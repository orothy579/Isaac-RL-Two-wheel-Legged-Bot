#!/usr/bin/env python
"""Fixed-height jump capability probe — "can the robot get onto an H-cm step at all?"

Runs one training per step height (default 0.10..0.15 m, 1 cm apart) on stairs whose risers
are ALL exactly that height (no terrain-level curriculum), warm-started from the current
stair champion. The verdict per height is binary:

* PASS ("된다")  — a meaningful fraction of episodes end with the robot on the first tread
  (``Curriculum/climb_units`` tail mean >= pass_frac x one-step units).
* FAIL ("안된다") — the full iteration budget elapses without that ever happening.

The point is a capability frontier, NOT performance: a clear PASS stops the run early (the
budget is only burned on heights that never work), and nothing about the robot's action or
impulse limits is modified. ``climb_units`` counts height in the nominal 5 cm units of
``StairClimbProgress`` (one 0.1 m step = 2 units, one 0.15 m step = 3 units).

Usage (after activating env_isaaclab)::

    python scripts/co_rl/probe_jump.py                      # 10..15 cm, 5000 it each
    python scripts/co_rl/probe_jump.py --heights 0.15       # single height
    python scripts/co_rl/probe_jump.py --results_dir <dir>  # resume (skips done heights)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep  # reuse: REPO_ROOT, TRAIN_PY, parse_log_dir, extract_metric

TASK = "Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo"
ALGO = "ppo"
# champion = timing sweep #12 (final_mean 1.344): its overrides + its final checkpoint
CHAMPION_OVERRIDES = os.path.join(
    sweep.REPO_ROOT,
    "logs/co_rl/Flamingo_Light_Rough_Stair_Jump/ppo/_sweeps/stair_jump_timing_2026-07-15_10-43-58/sweep012_overrides.json",
)
CHAMPION_CKPT = os.path.join(
    sweep.REPO_ROOT,
    "logs/co_rl/Flamingo_Light_Rough_Stair_Jump/ppo/2026-07-18_07-11-27_sweep012/model_4999.pt",
)
NOMINAL_UNIT = 0.05  # StairClimbProgress nominal step unit [m]
CLIMB_TAG = "climb_units"  # Curriculum/climb_units (log-only term in the stair_jump cfg)


def one_step_units(height: float) -> int:
    """Units StairClimbProgress reports for standing on ONE riser of this height."""
    return max(1, round(height / NOMINAL_UNIT))


def build_overrides(height: float) -> dict:
    """Champion reward/HP settings on a fixed-height, curriculum-free stair terrain."""
    with open(CHAMPION_OVERRIDES) as f:
        overrides = json.load(f)
    # the terrain-level curriculum is removed entirely, so drop its param overrides first
    overrides = {k: v for k, v in overrides.items() if not k.startswith("env.curriculum.terrain_levels")}
    overrides["env.curriculum.terrain_levels"] = None
    # every riser on every tile is exactly `height`; never load stale cached geometry
    overrides["env.scene.terrain.terrain_generator.sub_terrains.stair_up.step_height_range"] = [height, height]
    overrides["env.scene.terrain.terrain_generator.use_cache"] = False
    return overrides


def read_climb_curve(logfile: str) -> tuple[str | None, list[int], list[float]]:
    """(log_dir, steps, values) of ``Curriculum/climb_units`` so far — empty until data exists."""
    from tensorboard.backend.event_processing import event_accumulator

    log_dir = sweep.parse_log_dir(logfile)
    if not log_dir or not os.path.isdir(log_dir):
        return None, [], []
    try:
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = [t for t in ea.Tags().get("scalars", []) if CLIMB_TAG in t]
        if not tags:
            return log_dir, [], []
        scal = ea.Scalars(sorted(tags, key=len)[0])
        return log_dir, [s.step for s in scal], [s.value for s in scal]
    except Exception:
        return log_dir, [], []


def _tail_mean(vals: list[float], frac: float) -> float | None:
    if not vals:
        return None
    k = max(1, int(round(len(vals) * frac)))
    return sum(vals[-k:]) / k


def _kill(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run_probe(height: float, args, results_dir: str) -> dict:
    cm = round(height * 100)
    run_name = f"probe_h{cm}"
    units = one_step_units(height)
    pass_thr = args.pass_frac * units
    early_thr = args.early_frac * units

    override_path = os.path.join(results_dir, f"{run_name}_overrides.json")
    with open(override_path, "w") as f:
        json.dump(build_overrides(height), f, indent=2)
    logfile = os.path.join(results_dir, f"{run_name}.log")

    cmd = [
        args.python, sweep.TRAIN_PY,
        "--task", TASK, "--algo", ALGO,
        "--max_iterations", str(args.iters),
        "--run_name", run_name,
        "--param_overrides", override_path,
        "--headless", "--num_envs", str(args.num_envs),
        "--adaptive_reward",
        "--warmstart_ckpt", args.warmstart_ckpt,
    ]
    print(f"[probe] h={cm}cm (1 step = {units} units, pass >= {pass_thr:.2f}, early-stop >= {early_thr:.2f})")
    print(f"[probe]   $ {' '.join(cmd)}", flush=True)

    early_stopped = False
    stalled = False
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=sweep.REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            last_step, last_advance = -1, time.time()
            while proc.poll() is None:
                time.sleep(args.poll_sec)
                _, steps, vals = read_climb_curve(logfile)
                if steps and steps[-1] > last_step:
                    last_step, last_advance = steps[-1], time.time()
                cur = _tail_mean(vals, 0.1)
                if cur is not None:
                    print(f"[probe]   {datetime.now():%F %T} h={cm}cm it={last_step} climb_units(recent)={cur:.3f}", flush=True)
                if cur is not None and cur >= early_thr:
                    print(f"[probe]   h={cm}cm clear PASS ({cur:.3f} >= {early_thr:.2f}) — stopping early", flush=True)
                    early_stopped = True
                    _kill(proc)
                    break
                # Isaac Sim sometimes hangs in app shutdown AFTER training completes — the
                # curve is done, so reap the process instead of waiting forever.
                if last_step >= args.iters - 2:
                    print(f"[probe]   h={cm}cm training done (it={last_step}) but process still up — reaping", flush=True)
                    _kill(proc)
                    break
                # no TB progress for stall_sec (covers startup crashes and mid-run hangs)
                if time.time() - last_advance > args.stall_sec:
                    print(f"[probe]   h={cm}cm no progress for {args.stall_sec}s (it={last_step}) — killing", flush=True)
                    stalled = True
                    _kill(proc)
                    break
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            raise

    log_dir, steps, vals = read_climb_curve(logfile)
    final_val = _tail_mean(vals, 0.2)
    max_val = max(vals) if vals else None
    ran_enough = bool(steps) and steps[-1] >= args.iters * 0.5
    if final_val is None or (stalled and not ran_enough and final_val < pass_thr):
        status = "error" if final_val is None else "stalled"  # not enough evidence for a FAIL
        verdict = "?"
    else:
        status = "early_stop" if early_stopped else ("stalled" if stalled else "completed")
        verdict = "PASS" if (final_val >= pass_thr or (early_stopped and final_val >= early_thr * 0.5)) else "FAIL"
    return {
        "height_m": height, "units_one_step": units, "verdict": verdict, "status": status,
        "climb_units_final": final_val, "climb_units_max": max_val,
        "pass_threshold": round(pass_thr, 3), "log_dir": log_dir or "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heights", nargs="+", type=float,
                   default=[0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
    p.add_argument("--iters", type=int, default=5000,
                   help="budget per height; a clear PASS stops early, only FAILs burn it all")
    p.add_argument("--num_envs", type=int, default=8192)
    p.add_argument("--warmstart_ckpt", default=CHAMPION_CKPT)
    p.add_argument("--pass_frac", type=float, default=0.10,
                   help="PASS if tail-mean climb_units >= pass_frac x one-step units")
    p.add_argument("--early_frac", type=float, default=0.50,
                   help="stop the run early once climb_units >= early_frac x one-step units")
    p.add_argument("--poll_sec", type=int, default=600)
    p.add_argument("--stall_sec", type=int, default=2400,
                   help="kill a run whose TB curve hasn't advanced for this long (startup crash / hang)")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--results_dir", default=None, help="existing dir to resume (skips finished heights)")
    args = p.parse_args()

    if not os.path.isfile(args.warmstart_ckpt):
        sys.exit(f"[probe] FATAL: warmstart ckpt not found: {args.warmstart_ckpt}")
    if not os.path.isfile(CHAMPION_OVERRIDES):
        sys.exit(f"[probe] FATAL: champion overrides not found: {CHAMPION_OVERRIDES}")

    results_dir = args.results_dir or os.path.join(
        sweep.REPO_ROOT, "logs", "co_rl", "Flamingo_Light_Rough_Stair_Jump", ALGO,
        "_probe", f"probe_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "probe_results.csv")
    print(f"[probe] results dir: {results_dir}")

    done: dict[float, dict] = {}
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("verdict") in ("PASS", "FAIL"):
                    done[float(row["height_m"])] = row

    rows = list(done.values())
    for h in args.heights:
        if h in done:
            print(f"[probe] h={round(h*100)}cm already decided ({done[h]['verdict']}) — skipping")
            continue
        row = run_probe(h, args, results_dir)
        rows.append(row)
        rows.sort(key=lambda r: float(r["height_m"]))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n[probe] ===== capability verdicts =====")
    print(f"{'height':>7} {'verdict':>8} {'climb_units (tail/max)':>24} {'status':>11}")
    for r in sorted(rows, key=lambda r: float(r["height_m"])):
        fv = r["climb_units_final"]
        mv = r["climb_units_max"]
        fv = f"{float(fv):.3f}" if fv not in (None, "", "None") else "-"
        mv = f"{float(mv):.3f}" if mv not in (None, "", "None") else "-"
        print(f"{round(float(r['height_m'])*100):>5}cm {r['verdict']:>8} {fv:>11}/{mv:>11} {r['status']:>11}")
    print(f"[probe] full table: {csv_path}")


if __name__ == "__main__":
    main()
