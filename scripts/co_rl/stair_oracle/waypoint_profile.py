"""Simulator-independent, time-indexed joint waypoint format (SI units)."""

import bisect
import json
import math
from pathlib import Path

JOINTS = ("left_shoulder_joint", "right_shoulder_joint", "left_wheel_joint", "right_wheel_joint")
TORQUE_LIMIT = 17.0
SPEED_LIMIT = 40.0


def load_profile(path):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_profile(profile)
    return profile


def validate_profile(p):
    def number(v, label):
        if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
            raise ValueError(f"{label}: expected a finite number")
        return v

    if p.get("version") != 1:
        raise ValueError("profile version must be 1")
    s = p["scene"]
    for k in ("height_m", "tread_m", "width_m", "distance_m", "yaw_deg", "initial_base_z_m", "initial_vx_mps", "friction"):
        number(s[k], k)
    if s["height_m"] != 0.15 or s["tread_m"] != 0.30 or s["yaw_deg"] != 0:
        raise ValueError("v1 supports exactly 0.15 m riser, 0.30 m tread and head-on yaw=0 degrees")
    if min(s["width_m"], s["distance_m"], s["initial_base_z_m"]) <= 0 or s["friction"] < 0:
        raise ValueError("width/distance/base height must be positive; friction nonnegative")
    if not 0 <= s["initial_vx_mps"] <= 2:
        raise ValueError("initial_vx_mps must be in [0, 2]; it is an initial condition, not ongoing assistance")
    duration = number(p["duration_s"], "duration_s")
    if not 0 < duration <= 30:
        raise ValueError("duration_s must be in (0, 30]")
    w = p["waypoints"]
    if len(w) < 2 or w[0]["t"] != 0 or w[-1]["t"] > duration:
        raise ValueError("need at least two waypoints starting at t=0 and ending by duration_s")
    previous = -1
    for i, row in enumerate(w):
        t = number(row["t"], f"waypoint {i} time")
        if t <= previous:
            raise ValueError("waypoint times must be strictly increasing")
        previous = t
        for key, n in (("shoulder_rad", 2), ("wheel_rad_s", 2), ("feedforward_nm", 4)):
            if len(row[key]) != n:
                raise ValueError(f"waypoint {i}: {key} needs {n} entries")
            for value in row[key]:
                number(value, key)
        if any(abs(v) > SPEED_LIMIT for v in row["wheel_rad_s"]):
            raise ValueError("wheel target exceeds verified 40 rad/s limit")
        if any(abs(v) > TORQUE_LIMIT for v in row["feedforward_nm"]):
            raise ValueError("feedforward target exceeds verified 17 Nm limit")
        if len(row["base_reference"]) != 3:
            raise ValueError("base_reference is [x_m, z_m, pitch_rad], for comparison only")
        for v in row["base_reference"]:
            number(v, "base_reference")
    # Quintic smoothstep peak derivative = 1.875 / segment duration.
    for a, b in zip(w, w[1:]):
        if any(1.875 * abs(q1 - q0) / (b["t"] - a["t"]) > SPEED_LIMIT
               for q0, q1 in zip(a["shoulder_rad"], b["shoulder_rad"])):
            raise ValueError("shoulder interpolation exceeds verified 40 rad/s limit; increase segment duration")


def sample(p, t):
    """Quintic interpolation with zero joint target velocity at every keyframe."""
    w = p["waypoints"]
    i = max(0, min(len(w) - 2, bisect.bisect_right([r["t"] for r in w], t) - 1))
    a, b = w[i], w[i + 1]
    dt = b["t"] - a["t"]
    u = min(1., max(0., (t - a["t"]) / dt))
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = 30*u**2*(1-u)**2 / dt
    result = {key: [v0 + s*(v1-v0) for v0, v1 in zip(a[key], b[key])]
              for key in ("shoulder_rad", "wheel_rad_s", "feedforward_nm", "base_reference")}
    result["shoulder_rad_s"] = [ds*(q1-q0) for q0, q1 in zip(a["shoulder_rad"], b["shoulder_rad"])]
    result["phase"] = a.get("label", str(i)) if t < b["t"] else b.get("label", str(i+1))
    return result
