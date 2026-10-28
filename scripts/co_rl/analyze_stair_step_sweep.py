# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize a stair_step (alternating feet, ZMP) sweep from its TensorBoard logs.

For every trial of a sweep results dir (``sweepNNN.log`` + overrides) it reads the run's TB
scalars and reports the final value (mean of the last ``--tail`` fraction) of the gait / ZMP
metrics, the termination mix, the L4 main effects of each varied factor, and plots the curves.

    ./run_gb10.sh scripts/co_rl/analyze_stair_step_sweep.py \\
        logs/co_rl/Flamingo_Light_Rough_Stair_Step/ppo/_sweeps/<sweep dir> [--tail 0.2]
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

METRICS = {
    "terrain_level": "Curriculum/terrain_levels",
    "risers_climbed": "Curriculum/gait/risers_climbed",
    "climb_height_m": "Curriculum/gait/climb_height",
    "step_ups": "Curriculum/gait/step_ups",
    "alternating_ratio": "Curriculum/gait/alternating_ratio",
    "single_support_frac": "Curriculum/gait/single_support_frac",
    "flight_frac": "Curriculum/gait/flight_frac",
    "zmp_margin_m": "Curriculum/gait/zmp_margin",
    "episode_length": "Train/mean_episode_length",
    "noise_std": "Policy/mean_noise_std",
    # strict detector (leg lifted, no edge contact, landed on a tread); runs started after 2026-10-27 only
    "lifted_step_ups": "Curriculum/gait/lifted_step_ups",
    "lifted_share": "Curriculum/gait/lifted_share",
    "lifted_alternating_ratio": "Curriculum/gait/lifted_alternating_ratio",
}
TERMINATIONS = ("time_out", "no_progress", "base_contact", "bad_orientation", "shoulder_lower_violation")


def load_scalars(run_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing import event_accumulator as ea

    acc = ea.EventAccumulator(run_dir, size_guidance={ea.SCALARS: 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        s = acc.Scalars(tag)
        out[tag] = (np.array([x.step for x in s]), np.array([x.value for x in s]))
    return out


def find(scalars: dict, key: str):
    """Exact tag, else the shortest tag that contains ``key`` (``Episode/`` prefixes etc.)."""
    if key in scalars:
        return scalars[key]
    hits = sorted((t for t in scalars if key in t), key=len)
    return scalars[hits[0]] if hits else None


def tail_mean(series, frac: float) -> float:
    if series is None or len(series[1]) == 0:
        return float("nan")
    v = series[1]
    k = max(1, int(round(len(v) * frac)))
    return float(np.mean(v[-k:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    ap.add_argument("--tail", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="summary JSON (default <sweep_dir>/summary.json)")
    args = ap.parse_args()

    trials = []
    for fn in sorted(os.listdir(args.sweep_dir)):
        m = re.fullmatch(r"sweep(\d{3})\.log", fn)
        if not m:
            continue
        idx = int(m.group(1))
        log_dir = None
        warm = None
        with open(os.path.join(args.sweep_dir, fn), errors="ignore") as f:
            for line in f:
                if "Exact experiment name requested from command line:" in line:
                    log_dir = line.split("command line:", 1)[1].strip()
                if warm is None and "Warm-starting (transfer) from:" in line:
                    warm = line.split("from:", 1)[1].strip()
        params = json.load(open(os.path.join(args.sweep_dir, f"sweep{idx:03d}_overrides.json")))
        if log_dir is None or not os.path.isdir(log_dir):
            continue
        # perceptive variant = the policy group has the feet_scan term (recorded in the run's env.yaml)
        env_yaml = os.path.join(log_dir, "params", "env.yaml")
        perceptive = os.path.isfile(env_yaml) and "feet_scan:" in open(env_yaml, errors="ignore").read()
        # warm-start source (a per-trial sweep key, not a cfg override): the stand-drive or a stair_step run
        source = "none" if warm is None else ("stand_drive" if "Stand_Drive" in warm else "stair_step")
        trials.append({"trial": idx, "log_dir": log_dir, "params": params,
                       "observation": "perceptive" if perceptive else "blind", "warm_start": source})

    rows = []
    curves = {}
    for t in trials:
        sc = load_scalars(t["log_dir"])
        row = {"trial": t["trial"], "observation": t["observation"],
               "zmp_model": t["params"].get("env.zmp_gait.zmp_model"),
               "zmp_shaping": "high" if t["params"].get("env.rewards.zmp_tracking.weight", 1.0) > 1.5 else "low",
               "warm_start": t["warm_start"]}
        it = sc.get("Loss/learning_rate", (np.array([0]), None))[0]
        row["iterations"] = int(it[-1]) + 1 if len(it) else 0
        for name, tag in METRICS.items():
            row[name] = tail_mean(find(sc, tag), args.tail)
        for term in TERMINATIONS:
            row[f"term_{term}"] = tail_mean(find(sc, f"Episode_Termination/{term}"), args.tail)
        rows.append(row)
        curves[t["trial"]] = {k: find(sc, v) for k, v in METRICS.items()}

    # L4 main effects (each factor: mean(level b) - mean(level a) over the trials at each level)
    effects = {}
    for factor in ("observation", "zmp_model", "zmp_shaping", "warm_start"):
        levels = sorted({r[factor] for r in rows})
        if len(levels) != 2:
            continue
        a, b = levels
        effects[f"{factor}: {b} - {a}"] = {
            m: float(np.nanmean([r[m] for r in rows if r[factor] == b]) - np.nanmean([r[m] for r in rows if r[factor] == a]))
            for m in ("terrain_level", "risers_climbed", "step_ups", "alternating_ratio", "single_support_frac",
                      "lifted_step_ups", "lifted_share")
        }

    summary = {"tail_frac": args.tail, "trials": rows, "main_effects": effects}
    out = args.out or os.path.join(args.sweep_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    cols = ["trial", "observation", "zmp_model", "zmp_shaping", "warm_start", "iterations", "terrain_level",
            "risers_climbed", "step_ups", "alternating_ratio", "lifted_step_ups", "lifted_share",
            "single_support_frac", "flight_frac", "zmp_margin_m", "episode_length",
            "term_no_progress", "term_bad_orientation", "term_base_contact", "term_time_out"]
    print("\t".join(cols))
    for r in sorted(rows, key=lambda r: -np.nan_to_num(r["terrain_level"], nan=-1)):
        print("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    print(json.dumps(effects, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["terrain_level", "risers_climbed", "step_ups", "alternating_ratio", "single_support_frac", "noise_std",
                 "lifted_step_ups", "lifted_share", "lifted_alternating_ratio"]
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        for ax, name in zip(axes.flat, names):
            for r in rows:
                s = curves[r["trial"]].get(name)
                if s is None:
                    continue
                label = f"#{r['trial']} {r['observation']}/{r['zmp_model']}/{r['zmp_shaping']}/{r['warm_start']}"
                ax.plot(s[0], s[1], label=label, lw=1)
            ax.set_title(name)
            ax.set_xlabel("iteration")
            ax.grid(alpha=0.3)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(args.sweep_dir, "curves.png")
        fig.savefig(png, dpi=110)
        print(f"[analyze] curves -> {png}")
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] plot skipped: {e}")
    print(f"[analyze] summary -> {out}")


if __name__ == "__main__":
    main()
