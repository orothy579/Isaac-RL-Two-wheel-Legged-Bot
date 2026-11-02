"""Score a stair-step policy (perceptive or BLIND) in the MuJoCo cosim of v01_2_2 with the cosim's own evaluator.

``cosim-anaylze/tools/eval_stair_step_v01_2_2.py`` builds the perceptive policy input (3 frames + command + 72-ray
scan). The blind student sees 50 frames + command and no scan. This runs that evaluator's ``run_trial`` unchanged -
robot, stairs, actuators, randomisation, terminations and the pass rule - and only swaps ``Sim.observe`` for the
layout given by ``--frames`` / ``--no_scan``, and adds Isaac's joint and link speed limits around ``mj_step``
(PhysX enforces them, MuJoCo has none; ``--no_speed_limits`` drops them). The report is the evaluator's, plus ``checkpoint`` (the co_rl .pt the
actor was exported from) and ``"sim": "mujoco"``, so ``scripts/co_rl/sweeps/pick_best_blind.py`` can combine it with
the Isaac evaluation of the same checkpoint.

    CUDA_VISIBLE_DEVICES= <venv python> cosim-anaylze/tools/export_co_rl_actor.py <model.pt> <actor.npz>
    <cosim_venv python> scripts/co_rl/cosim_eval_policy.py --actor <actor.npz> --checkpoint <model.pt> \\
        --frames 50 --no_scan --levels 0 3 6 9 --trials 256 --workers 8 --out <json>
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../cosim-anaylze/tools")))
import eval_stair_step_v01_2_2 as ev  # noqa: E402

# read at import: the spawned pool workers import this module and get the same layout
FRAMES = int(os.environ.get("COSIM_EVAL_FRAMES", "3"))
SCAN = os.environ.get("COSIM_EVAL_SCAN", "1") == "1"


# COSIM_EVAL_OBS_NOISE=1: Isaac's training noise on each new frame (uniform, added before the obs scale there):
# shoulder pos +-0.05, joint vel +-1.5 rad/s x 0.15, base ang vel +-0.15 rad/s x 0.25, gravity +-0.05, action +-0.01
OBS_NOISE = os.environ.get("COSIM_EVAL_OBS_NOISE", "0") == "1"
NOISE_HALF = np.array([0.05] * 2 + [1.5 * 0.15] * 4 + [0.15 * 0.25] * 3 + [0.05] * 3 + [0.01] * 4)
_noise_rng = np.random.default_rng(os.getpid())


def _observe(self, command: float) -> np.ndarray:
    """``stack_policy`` frames (newest first), the scaled command, then the feet scan if the policy sees it."""
    f = self.frame()
    if OBS_NOISE:
        f = f + _noise_rng.uniform(-NOISE_HALF, NOISE_HALF)
    self.frames = [f] * FRAMES if not self.frames else [f] + self.frames[: FRAMES - 1]
    parts = self.frames + [np.array([2.0 * command, 0.0, 0.0, 0.0])]
    if SCAN:
        parts.append(self.scan())
    return np.concatenate(parts)


ev.Sim.observe = _observe

# --max_speed: cap the wheel speed targets like Isaac's action clip - the target is clipped, the observed last
# action stays the policy's own output
MAX_WHEEL = float(os.environ.get("COSIM_EVAL_MAX_WHEEL", "inf"))
_orig_step = ev.Sim.step


def _step(self, a: np.ndarray):
    if not np.isfinite(MAX_WHEEL):
        return _orig_step(self, a)
    lim = MAX_WHEEL / ev.WHEEL_SCALE
    clipped = a.copy()
    clipped[2:] = np.clip(clipped[2:], -lim, lim)
    _orig_step(self, clipped)
    self.last_action = a


ev.Sim.step = _step

# Isaac's speed limits, which MuJoCo does not have (on unless --no_speed_limits). Measured in Isaac with the base
# fixed and the wheels spun in the air:
# 1. joint speed: the USD's physxJoint:maxJointVelocity, a hard clamp - 40 rad/s on the shoulder, upper, leg and
#    wheel joints, 100 rad/s on the casters
# 2. link angular speed: the spawn cfg's rigid-body max_angular_velocity, 1000 deg/s (17.45 rad/s) on every link.
#    PhysX pulls a faster link back before each 5 ms step: a spinning wheel settles where the drive torque equals
#    I_link (|w| - 17.45) / 5 ms (targets 25 / 35 / 60 rad/s -> 20.5 / 24.3 / 33.8 rad/s, below 17.45 no effect).
#    Applied here as that torque on every link above the limit.
SPEED_LIMITS = os.environ.get("COSIM_EVAL_SPEED_LIMITS", "1") == "1"
LINK_SPEED = np.radians(1000.0)


class _SpeedLimits:
    """One model's tables: hinge dofs and their speed limits; for every robot body, the hinges between it and the base."""

    def __init__(self, m):
        mj = ev.mujoco_unlimited
        hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        self.dofs = np.array([m.jnt_dofadr[j] for j in hinges])
        self.vmax = np.array([100.0 if "caster" in m.joint(j).name else 40.0 for j in hinges])
        # world angular velocity of body b = that of the floating base + the hinge rates along its chain:
        # w = w_base + chain @ (axis_j * qd_j), chain[b, j] = 1 for every hinge between the base and b
        self.nbody = m.nbody
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
        if len(free) > 1:
            raise ValueError(f"expected one floating base, found {len(free)}")
        # no free joint: a fixed base (tests), the chains run up to the world body
        self.base, self.base_dof = (m.jnt_bodyid[free[0]], m.jnt_dofadr[free[0]]) if free else (0, None)
        self.hinge_jnt = np.array(hinges)
        col = {j: k for k, j in enumerate(hinges)}
        ids, rows = [], []
        for b in range(1, m.nbody):
            if m.body_jntnum[b] > 1:
                raise ValueError(f"body {m.body(b).name} has {m.body_jntnum[b]} joints")
            row, x = np.zeros(len(hinges)), b
            while x > 0 and x != self.base:  # walk up to the base
                if m.body_jntnum[x] == 1 and m.body_jntadr[x] in col:
                    row[col[m.body_jntadr[x]]] = 1.0
                x = m.body_parentid[x]
            if x == self.base:  # the robot (no other bodies move)
                ids.append(b)
                rows.append(row)
        self.ids, self.chain = np.array(ids), np.array(rows)

    def link_torques(self, m, d) -> None:
        """World torque -I (w - w_max) / dt on every body spinning faster than LINK_SPEED (0 on the others)."""
        w = self.chain @ (d.xaxis[self.hinge_jnt] * d.qvel[self.dofs][:, None])
        if self.base_dof is not None:
            w += d.xmat[self.base].reshape(3, 3) @ d.qvel[self.base_dof + 3:self.base_dof + 6]
        d.xfrc_applied[self.ids, 3:] = 0.0
        speed = np.linalg.norm(w, axis=1)
        for r in np.nonzero(speed > LINK_SPEED)[0]:
            b = self.ids[r]
            R = d.ximat[b].reshape(3, 3)
            excess = w[r] * (1.0 - LINK_SPEED / speed[r])
            d.xfrc_applied[b, 3:] = -(R @ (m.body_inertia[b] * (R.T @ excess))) / ev.DT


class _MujocoWithSpeedLimits:
    """The evaluator's ``mujoco`` module with Isaac's speed limits around every ``mj_step``."""

    def __init__(self, mj):
        self._mj = mj
        self._limits = None

    def __getattr__(self, name):
        return getattr(self._mj, name)

    def mj_step(self, m, d, nstep: int = 1):
        # every model is the same robot on other stair boxes (static geoms): one table serves all
        if self._limits is None or self._limits.nbody != m.nbody:
            self._limits = _SpeedLimits(m)
        lim = self._limits
        for _ in range(nstep):
            lim.link_torques(m, d)
            self._mj.mj_step(m, d)
            d.qvel[lim.dofs] = np.clip(d.qvel[lim.dofs], -lim.vmax, lim.vmax)


ev.mujoco_unlimited = ev.mujoco
if SPEED_LIMITS:
    ev.mujoco = _MujocoWithSpeedLimits(ev.mujoco_unlimited)


def trial(args: tuple) -> dict:
    return ev.run_trial(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--actor", required=True, help=".npz from cosim-anaylze/tools/export_co_rl_actor.py")
    ap.add_argument("--checkpoint", default=None, help="the co_rl model_N.pt the actor came from (for the report)")
    ap.add_argument("--frames", type=int, default=3, help="stacked proprioceptive frames (blind student: 50)")
    ap.add_argument("--no_scan", action="store_true", help="blind policy: no feet scan in the input")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6, 9])
    ap.add_argument("--trials", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--command", type=float, default=0.35)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_speed", type=float, default=None,
                    help="also clip the wheel speed targets at this rim speed [m/s] (this cuts the drive torque too)")
    ap.add_argument("--no_speed_limits", action="store_true", help="without Isaac's joint / link speed limits")
    ap.add_argument("--obs_noise", action="store_true", help="add Isaac's training observation noise")
    a = ap.parse_args()
    if a.max_speed is not None:
        os.environ["COSIM_EVAL_MAX_WHEEL"] = str(a.max_speed / ev.WHEEL_RADIUS)
    os.environ["COSIM_EVAL_SPEED_LIMITS"] = "0" if a.no_speed_limits else "1"
    os.environ["COSIM_EVAL_OBS_NOISE"] = "1" if a.obs_noise else "0"
    os.environ["COSIM_EVAL_FRAMES"] = str(a.frames)
    os.environ["COSIM_EVAL_SCAN"] = "0" if a.no_scan else "1"
    n_in = ev.Actor(a.actor).layers[0][0].shape[1]
    expect = 16 * a.frames + 4 + (0 if a.no_scan else len(ev.SCAN_XY))
    if n_in != expect:
        sys.exit(f"actor input {n_in} != {expect} for {a.frames} frames {'without' if a.no_scan else 'with'} scan")
    jobs = [(lv, int(i // (a.trials / ev.COLS)), i, a.seed * 100003 + lv * 10007 + i, a.actor, a.seconds, a.command,
             a.yaw, False) for lv in a.levels for i in range(a.trials)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(trial, jobs, chunksize=4)
    report = {"sim": "mujoco", "checkpoint": a.checkpoint, "actor": a.actor, "frames": a.frames, "scan": not a.no_scan,
              "max_speed": a.max_speed, "isaac_speed_limits": not a.no_speed_limits, "obs_noise": a.obs_noise,
              "trials": a.trials, "seconds": a.seconds, "command": a.command, "yaw": a.yaw, "seed": a.seed,
              "mujoco_version": ev.mujoco.__version__,
              "actor_sha256": hashlib.sha256(open(a.actor, "rb").read()).hexdigest(),
              "robot_xml_sha256": hashlib.sha256(open(ev.ROBOT_XML, "rb").read()).hexdigest(),
              "evaluator_sha256": hashlib.sha256(open(ev.__file__, "rb").read()).hexdigest(),
              "wall_s": round(time.time() - t0, 1), "levels": {}}
    for lv in a.levels:
        r = [x for x in res if x["level"] == lv]
        risers = np.array([x["risers"] for x in r])
        ends, firsts = {}, {}
        for x in r:
            ends[x["end"]] = ends.get(x["end"], 0) + 1
            firsts[x["first"]] = firsts.get(x["first"], 0) + 1
        report["levels"][str(lv)] = {
            "riser_height_m": round(ev.nominal_riser(lv), 4),
            "envs_climbed_1plus": round(float((risers >= 0.9).mean()), 3),
            "envs_climbed_3plus": round(float((risers >= 2.9).mean()), 3),
            "envs_climbed_nplus": {str(k): round(float((risers >= k - 0.1).mean()), 3) for k in (1, 2, 3, 4, 5)},
            "risers_climbed_mean": round(float(risers.mean()), 3),
            "mean_episode_s": round(float(np.mean([x["t_end"] for x in r])), 3),
            "first_wheel": firsts, "terminations": ends}
        print(f"[cosim] level {lv} ({ev.nominal_riser(lv) * 100:.1f} cm): 3 risers {(risers >= 2.9).mean() * 100:.1f} %, "
              f"mean {risers.mean():.2f}, >=1 {(risers >= 0.9).mean() * 100:.1f} %, ends {ends}", flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
