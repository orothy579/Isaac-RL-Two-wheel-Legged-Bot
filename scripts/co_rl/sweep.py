# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reward / hyperparameter sweep driver for CO-RL training.

Reads a YAML config that declares, per parameter, a search range (grid step, an explicit
value list, or a continuous min/max for random sampling), then launches one ``train.py``
subprocess per trial — each a full ``max_iterations`` run — injecting that trial's parameters
via ``--param_overrides``. After each run it reads a metric (e.g. ``Curriculum/terrain_levels``)
straight from the run's TensorBoard events, then ranks all trials.

This is the *grid / random search* layer (standard HPO). It is also the substrate for a future
Population-Based-Training loop (checkpoint + exploit/explore instead of independent runs).

Usage::

    python scripts/co_rl/sweep.py --config scripts/co_rl/sweeps/stair_jump_example.yaml
    python scripts/co_rl/sweep.py --config <cfg.yaml> --dry_run   # just list the trials

Config schema (see ``sweeps/stair_jump_example.yaml``)::

    task: Isaac-Velocity-Rough-Flamingo-Light-Stair-Jump-v1-ppo
    algo: ppo
    max_iterations: 3000
    method: grid            # grid | random | optuna (TPE; needs `pip install optuna`)
    num_samples: 20         # random / optuna: number of trials
    seed: 42
    base_args: ["--headless", "--num_envs", "4096", "--warmstart_ckpt", "<path>"]
    metric:
      tag_contains: terrain_level   # TB scalar tag substring
      reduce: last                  # last | max | mean | auc | iters_to
      threshold: 5.0                # iters_to only
      goal: max                     # max | min  (iters_to -> min)
    parameters:
      env.rewards.stair_climb.weight:        {min: 10, max: 30, step: 5}
      env.rewards.stair_climb.params.growth: {values: [1.5, 2.0, 2.5]}
      agent.algorithm.entropy_coef:          {min: 0.002, max: 0.02, log: true}
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_PY = os.path.join("scripts", "co_rl", "train.py")
LOG_DIR_MARKER = "Exact experiment name requested from command line:"


# --------------------------------------------------------------------------------------
# trial generation
# --------------------------------------------------------------------------------------
def _grid_points(spec: dict) -> list:
    """Discrete grid points for one parameter."""
    if "values" in spec:
        return list(spec["values"])
    if "min" in spec and "max" in spec and "step" in spec:
        lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
        n = int(round((hi - lo) / step))
        pts = [round(lo + i * step, 10) for i in range(n + 1)]
        return pts
    raise ValueError(f"grid parameter needs 'values' or 'min/max/step'; got {spec}")


def expand_grid(parameters: dict) -> list[dict]:
    """Cartesian product of every parameter's grid points."""
    names = list(parameters)
    axes = [_grid_points(parameters[n]) for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*axes)]


def _sample_one(spec: dict, rng: random.Random):
    if "values" in spec:
        return rng.choice(list(spec["values"]))
    lo, hi = float(spec["min"]), float(spec["max"])
    if spec.get("log"):
        val = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    else:
        val = rng.uniform(lo, hi)
    if "step" in spec:  # snap to grid if a step is given
        step = float(spec["step"])
        val = lo + round((val - lo) / step) * step
    return round(val, 10)


def sample_random(parameters: dict, n: int, rng: random.Random) -> list[dict]:
    return [{name: _sample_one(spec, rng) for name, spec in parameters.items()} for _ in range(n)]


def suggest_params(trial, parameters: dict) -> dict:
    """Map the SAME YAML parameter spec onto Optuna ``trial.suggest_*`` calls (TPE search)."""
    out = {}
    for name, spec in parameters.items():
        if "values" in spec:
            out[name] = trial.suggest_categorical(name, list(spec["values"]))
        elif "min" in spec and "max" in spec:
            lo, hi = float(spec["min"]), float(spec["max"])
            if spec.get("log"):
                out[name] = trial.suggest_float(name, lo, hi, log=True)
            elif "step" in spec:
                out[name] = trial.suggest_float(name, lo, hi, step=float(spec["step"]))
            else:
                out[name] = trial.suggest_float(name, lo, hi)
        else:
            raise ValueError(f"parameter '{name}' needs 'values' or 'min/max'; got {spec}")
    return out


# --------------------------------------------------------------------------------------
# running a trial + reading its metric
# --------------------------------------------------------------------------------------
def parse_log_dir(logfile: str) -> str | None:
    """Recover the run's log dir from the marker line train.py prints."""
    log_dir = None
    try:
        with open(logfile) as f:
            for line in f:
                if LOG_DIR_MARKER in line:
                    # A resumed grid trial appends to the same sweep log. Keep the LAST
                    # marker so metric extraction uses the resumed run, not the interrupted
                    # run whose marker appears first.
                    log_dir = line.split(LOG_DIR_MARKER, 1)[1].strip()
    except OSError:
        return None
    return log_dir


def extract_metric(log_dir: str, metric: dict) -> float:
    """Reduce a TensorBoard scalar (tag containing ``tag_contains``) to a single number."""
    from tensorboard.backend.event_processing import event_accumulator

    # find the dir that actually holds the event file (usually log_dir itself)
    ev_dir = log_dir
    if not any(f.startswith("events.out.tfevents") for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))):
        for root, _, files in os.walk(log_dir):
            if any(f.startswith("events.out.tfevents") for f in files):
                ev_dir = root
                break

    ea = event_accumulator.EventAccumulator(ev_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    want = metric.get("tag_contains", "terrain_level")
    matches = [t for t in tags if want in t]
    if not matches:
        raise KeyError(f"no scalar tag contains '{want}'. available: {tags}")
    tag = sorted(matches, key=len)[0]  # shortest match = the plain one
    scal = ea.Scalars(tag)
    steps = [s.step for s in scal]
    vals = [s.value for s in scal]
    if not vals:
        raise KeyError(f"tag '{tag}' has no data")

    reduce = metric.get("reduce", "last")
    if reduce == "last":
        return float(vals[-1])
    if reduce == "max":
        return float(max(vals))
    if reduce == "mean":
        return float(sum(vals) / len(vals))
    if reduce == "final_mean":  # mean of the last tail_frac of the curve = final achieved level,
        # robust to single-episode spikes. Prefers runs that REACH & HOLD high terrain (unlike
        # auc, which over-rewards reaching it early). Use this when peak/final level is the goal.
        tail = float(metric.get("tail_frac", 0.2))
        k = max(1, int(round(len(vals) * tail)))
        return float(sum(vals[-k:]) / k)
    if reduce == "auc":  # area under the curve: rewards reaching high AND early
        import numpy as np

        trapz = getattr(np, "trapezoid", None) or np.trapz  # np 2.0 renamed trapz -> trapezoid
        return float(trapz(vals, steps))
    if reduce == "iters_to":  # first step to cross threshold (fewer = faster); never -> penalty
        thr = float(metric.get("threshold", 0.0))
        for st, v in zip(steps, vals):
            if v >= thr:
                return float(st)
        return float(steps[-1] * 2 if steps else 1e9)
    raise ValueError(f"unknown reduce '{reduce}'")


def _find_previous_run(cfg: dict, run_name: str) -> str | None:
    """Find a previous run directory matching *run_name* (e.g. ``sweep012``) that contains
    at least one ``model_*.pt`` checkpoint.

    Returns the *run-folder basename* (e.g. ``2026-07-08_19-55-35_sweep012``) or ``None``.
    """
    exp_name = cfg.get("experiment_name")
    algo = cfg.get("algo", "ppo")
    if exp_name:
        run_root = os.path.join(REPO_ROOT, "logs", "co_rl", exp_name, algo)
    else:
        return None
    if not os.path.isdir(run_root):
        return None
    # run folders end with _{run_name}, e.g. 2026-07-08_19-55-35_sweep012
    candidates = [d for d in os.listdir(run_root)
                  if d.endswith(f"_{run_name}") and os.path.isdir(os.path.join(run_root, d))]
    if not candidates:
        return None
    # pick the most recent one (lexicographic = chronological for the timestamp prefix)
    candidates.sort()
    for cand in reversed(candidates):
        cand_path = os.path.join(run_root, cand)
        ckpts = [f for f in os.listdir(cand_path) if f.startswith("model_") and f.endswith(".pt")]
        if ckpts:
            return cand
    return None


def _latest_checkpoint(run_dir: str) -> str | None:
    """Return the filename of the highest-iteration ``model_*.pt`` in *run_dir*."""
    ckpts = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not ckpts:
        return None
    # model_2700.pt -> 2700
    def _iter_num(f):
        try:
            return int(f.replace("model_", "").replace(".pt", ""))
        except ValueError:
            return -1
    return max(ckpts, key=_iter_num)


def _checkpoint_iteration(filename: str) -> int:
    """Parse ``model_<iteration>.pt`` and fail loudly on an unexpected filename."""
    try:
        return int(filename.removeprefix("model_").removesuffix(".pt"))
    except ValueError as exc:
        raise ValueError(f"invalid checkpoint filename: {filename!r}") from exc


def build_cmd(python: str, task: str, algo: str, max_iter: int, run_name: str,
              override_path: str, base_args: list[str],
              resume_run: str | None = None, resume_ckpt: str | None = None) -> list[str]:
    cmd = [
        python, TRAIN_PY,
        "--task", task, "--algo", algo,
        "--max_iterations", str(max_iter),
        "--run_name", run_name,
        "--param_overrides", override_path,
    ]
    if resume_run and resume_ckpt:
        cmd += ["--resume", "True", "--load_run", resume_run, "--checkpoint", resume_ckpt]
        # remove --warmstart_ckpt from base_args when resuming (resume takes precedence)
        filtered = []
        skip_next = False
        for a in base_args:
            if skip_next:
                skip_next = False
                continue
            if str(a) == "--warmstart_ckpt":
                skip_next = True
                continue
            filtered.append(str(a))
        cmd += filtered
    else:
        cmd += [str(a) for a in base_args]
    return cmd


def run_trial(idx: int, params: dict, cfg: dict, results_dir: str, python: str,
              resume: bool = False) -> dict:
    run_name = f"sweep{idx:03d}"
    override_path = os.path.join(results_dir, f"{run_name}_overrides.json")
    with open(override_path, "w") as f:
        json.dump(params, f, indent=2)
    logfile = os.path.join(results_dir, f"{run_name}.log")

    # --- checkpoint resume support ---
    resume_run = None
    resume_ckpt = None
    run_iterations = int(cfg["max_iterations"])
    if resume:
        # Prefer the exact run recorded in this sweep's own log. Falling back to a global
        # suffix search can accidentally resume an unrelated, newer ``*_sweep000`` run.
        previous_log_dir = parse_log_dir(logfile)
        prev_run = None
        if previous_log_dir and os.path.isdir(previous_log_dir):
            prev_run = os.path.basename(previous_log_dir)
        if prev_run is None:
            prev_run = _find_previous_run(cfg, run_name)
        if prev_run:
            exp_name = cfg.get("experiment_name")
            algo = cfg.get("algo", "ppo")
            prev_dir = os.path.join(REPO_ROOT, "logs", "co_rl", exp_name, algo, prev_run)
            ckpt = _latest_checkpoint(prev_dir)
            if ckpt:
                resume_run = prev_run
                resume_ckpt = ckpt
                ckpt_iter = _checkpoint_iteration(ckpt)
                # ``OnPolicyRunner.learn(n)`` interprets n as ADDITIONAL iterations after
                # loading. Its range starts at ckpt_iter (repeating that boundary update),
                # hence target - ckpt_iter reaches model_(target-1) exactly.
                run_iterations = max(1, int(cfg["max_iterations"]) - ckpt_iter)
                print(
                    f"[sweep] trial {idx}: RESUMING from {prev_run}/{ckpt} "
                    f"for {run_iterations} remaining iterations"
                )
            else:
                print(f"[sweep] trial {idx}: previous run {prev_run} found but no checkpoints; starting fresh")
        else:
            print(f"[sweep] trial {idx}: no previous run found for resume; starting fresh")

    cmd = build_cmd(python, cfg["task"], cfg.get("algo", "ppo"), run_iterations,
                    run_name, override_path, cfg.get("base_args", []),
                    resume_run=resume_run, resume_ckpt=resume_ckpt)
    print(f"[sweep] trial {idx}: {params}")
    print(f"[sweep]   $ {' '.join(cmd)}")
    with open(logfile, "w" if not resume_run else "a") as lf:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT, env=os.environ)

    row = {"trial": idx, **params, "status": "ok", "metric": float("nan"), "log_dir": None}
    if proc.returncode != 0:
        row["status"] = f"train_exit_{proc.returncode}"
        print(f"[sweep]   FAILED (exit {proc.returncode}); see {logfile}")
        return row
    log_dir = parse_log_dir(logfile)
    row["log_dir"] = log_dir
    if not log_dir or not os.path.isdir(log_dir):
        row["status"] = "no_log_dir"
        return row
    try:
        row["metric"] = extract_metric(log_dir, cfg.get("metric", {}))
    except Exception as e:  # noqa: BLE001 - record and continue the sweep
        row["status"] = f"metric_error: {e}"
    print(f"[sweep]   metric={row['metric']}  status={row['status']}")
    return row


def _mirror_to_run(row: dict, results_dir: str) -> None:
    """Copy this trial's sweep files into its run folder's ``sweep/`` subdir, so each run
    folder is self-contained (its overrides, log, and the running ranking live right next
    to the trained model)."""
    log_dir = row.get("log_dir")
    if not log_dir or not os.path.isdir(log_dir):
        return
    dst = os.path.join(log_dir, "sweep")
    os.makedirs(dst, exist_ok=True)
    idx = row["trial"]
    for fn in (f"sweep{idx:03d}_overrides.json", f"sweep{idx:03d}.log", "results.csv", "results_ranked.csv"):
        src = os.path.join(results_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, fn))


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CO-RL reward/hyperparameter sweep driver.")
    p.add_argument("--config", required=True, help="Path to the sweep YAML config.")
    p.add_argument("--dry_run", action="store_true", help="List the trials and exit (no training).")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for train.py subprocesses.")
    p.add_argument("--results_dir", default=None, help="Where to write trial logs/overrides/results.")
    p.add_argument("--seed_from", default=None,
                   help="Path to an old study.db whose COMPLETE trials are imported into this (fresh) "
                        "study, so a broad sweep continues from earlier results instead of from zero.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    method = cfg.get("method", "grid")
    goal = cfg.get("metric", {}).get("goal", "max")
    print(f"[sweep] method={method}  task={cfg['task']}  max_iterations={cfg['max_iterations']}  "
          f"metric={cfg.get('metric', {})}")

    if args.dry_run:
        if method == "optuna":
            print("[sweep] optuna (TPE) search space:")
            for n, s in cfg["parameters"].items():
                print(f"  {n}: {s}")
            print(f"[sweep] dry run — would run {cfg.get('num_samples', 20)} TPE trials.")
        else:
            rng = random.Random(cfg.get("seed", 42))
            trials = expand_grid(cfg["parameters"]) if method == "grid" \
                else sample_random(cfg["parameters"], int(cfg.get("num_samples", 10)), rng)
            for i, t in enumerate(trials):
                print(f"  trial {i:3d}: {t}")
            print(f"[sweep] dry run — {len(trials)} trials would run (~{cfg['max_iterations']} iters each).")
        return

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = os.path.splitext(os.path.basename(args.config))[0]
    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
    elif cfg.get("experiment_name"):
        # co-locate the sweep with the task's runs: logs/co_rl/<exp>/<algo>/_sweeps/<name>_<stamp>
        # (so results.csv / study.db live right next to the per-trial run folders, not off in a
        # separate _sweeps tree — easier to manage).
        results_dir = os.path.join(REPO_ROOT, "logs", "co_rl", cfg["experiment_name"],
                                   cfg.get("algo", "ppo"), "_sweeps", f"{name}_{stamp}")
    else:
        results_dir = os.path.join(REPO_ROOT, "logs", "co_rl", "_sweeps", f"{name}_{stamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[sweep] results dir: {results_dir}")

    if method == "optuna":
        _run_optuna(cfg, results_dir, args.python, goal, args.seed_from)
        return
    if method not in ("grid", "random"):
        raise ValueError(f"method must be 'grid', 'random', or 'optuna', got '{method}'")

    rng = random.Random(cfg.get("seed", 42))
    trials = expand_grid(cfg["parameters"]) if method == "grid" \
        else sample_random(cfg["parameters"], int(cfg.get("num_samples", 10)), rng)
    # Grid/random sweeps are sequential and can be interrupted in the middle of a long
    # training run. Preserve already completed rows and resume an incomplete trial from
    # the checkpoint referenced by this exact results directory.
    rows = []
    completed: dict[int, dict] = {}
    results_csv = os.path.join(results_dir, "results.csv")
    if os.path.isfile(results_csv):
        import pandas as pd

        for row in pd.read_csv(results_csv).to_dict(orient="records"):
            if str(row.get("status")) != "ok":
                continue
            # A SIGINT can make train.py exit cleanly after saving an intermediate model.
            # Do not trust status alone: a trial is complete only if its latest checkpoint
            # reached the configured final iteration.
            row_run = row.get("log_dir")
            ckpt = _latest_checkpoint(row_run) if isinstance(row_run, str) and os.path.isdir(row_run) else None
            ckpt_iter = _checkpoint_iteration(ckpt) if ckpt else -1
            if ckpt_iter >= int(cfg["max_iterations"]) - 1:
                completed[int(row["trial"])] = row
            else:
                print(
                    f"[sweep] trial {int(row['trial'])}: status=ok but latest checkpoint is "
                    f"{ckpt or 'missing'} (< model_{int(cfg['max_iterations']) - 1}.pt); resuming"
                )
        if completed:
            print(f"[sweep] reusing {len(completed)} completed grid trial(s) from results.csv")
    try:
        for i, t in enumerate(trials):
            if i in completed:
                rows.append(completed[i])
                print(f"[sweep] trial {i}: already complete — skipping")
                continue

            trial_log = os.path.join(results_dir, f"sweep{i:03d}.log")
            previous_log_dir = parse_log_dir(trial_log)
            has_checkpoint = False
            if previous_log_dir and os.path.isdir(previous_log_dir):
                has_checkpoint = _latest_checkpoint(previous_log_dir) is not None
            rows.append(run_trial(i, t, cfg, results_dir, args.python, resume=has_checkpoint))
            _write_results(rows, results_dir, goal)  # checkpoint after every trial
            _mirror_to_run(rows[-1], results_dir)    # also drop results into the run folder's sweep/
            if rows[-1]["status"].startswith("train_exit") or rows[-1]["status"] == "no_log_dir":
                print(f"[sweep] ABORTED at trial {i}: training didn't start (status={rows[-1]['status']}). "
                      "Check the log, fix the environment, then re-run.")
                break
    except KeyboardInterrupt:
        print("\n[sweep] interrupted — writing partial results.")

    best = _write_results(rows, results_dir, goal)
    print(f"\n[sweep] done. best trial: {best}")


def _run_optuna(cfg: dict, results_dir: str, python: str, goal: str, seed_from: str | None = None) -> None:
    """TPE-sampled Optuna study over the same parameter spec; resumable via a SQLite store.

    Reuses ``run_trial`` (subprocess train.py + ``--param_overrides``) and ``extract_metric``.
    A failed run (non-zero exit / missing metric) is reported as pruned so the study continues.

    On resume, FAIL'd trials that have checkpoints on disk are **retried first** (with
    ``--resume``), so a crashed run picks up from its latest ``model_*.pt`` instead of
    wasting the training budget.
    """
    import optuna

    rows: list[dict] = []

    def objective(trial):
        params = suggest_params(trial, cfg["parameters"])
        row = run_trial(trial.number, params, cfg, results_dir, python)
        rows.append(row)
        _write_results(rows, results_dir, goal)  # checkpoint after every trial
        _mirror_to_run(row, results_dir)          # also drop results into the run folder's sweep/
        status = row["status"]
        if status.startswith("train_exit") or status == "no_log_dir":
            # training never STARTED (env/command error, e.g. isaacsim not importable). Abort
            # instead of pruning — otherwise every remaining trial fails the same way in seconds
            # and silently burns the whole budget.
            raise RuntimeError(
                f"trial {trial.number} failed before training started (status={status}). See "
                f"{os.path.join(results_dir, f'sweep{trial.number:03d}.log')}. Fix the environment, "
                f"then re-run with --results_dir {results_dir} to resume."
            )
        if not math.isfinite(row["metric"]):
            raise optuna.TrialPruned()
        return row["metric"]

    study = optuna.create_study(
        study_name=os.path.basename(results_dir),
        direction="maximize" if goal == "max" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        storage=f"sqlite:///{os.path.join(results_dir, 'study.db')}",
        load_if_exists=True,
    )

    if seed_from and len(study.trials) == 0:  # warm-start a fresh study from an old one's results
        src = f"sqlite:///{os.path.abspath(seed_from)}"
        old = optuna.load_study(study_name=optuna.study.get_all_study_names(src)[0], storage=src)
        imported = 0
        for t in old.trials:
            if t.state.name == "COMPLETE" and t.value is not None:
                try:
                    study.add_trial(t)
                    imported += 1
                except Exception as e:  # noqa: BLE001 - distribution mismatch etc.
                    print(f"[sweep] skip seeding trial #{t.number}: {e}")
        print(f"[sweep] seeded {imported} completed trial(s) from {seed_from}")

    # ------------------------------------------------------------------
    # Phase 0: retry FAIL'd trials that have on-disk checkpoints
    # ------------------------------------------------------------------
    failed_trials = [t for t in study.trials if t.state.name == "FAIL"]
    resumable = []
    for ft in failed_trials:
        run_name = f"sweep{ft.number:03d}"
        prev_run = _find_previous_run(cfg, run_name)
        if prev_run:
            exp_name = cfg.get("experiment_name")
            algo = cfg.get("algo", "ppo")
            prev_dir = os.path.join(REPO_ROOT, "logs", "co_rl", exp_name, algo, prev_run)
            ckpt = _latest_checkpoint(prev_dir)
            if ckpt:
                resumable.append(ft)

    if resumable:
        print(f"[sweep] found {len(resumable)} FAIL'd trial(s) with checkpoints — retrying with --resume.")
    for ft in resumable:
        # reconstruct the params from the Optuna trial
        params = {k: v for k, v in ft.params.items()}
        print(f"[sweep] retrying trial {ft.number} from checkpoint...")
        row = run_trial(ft.number, params, cfg, results_dir, python, resume=True)
        rows.append(row)
        _write_results(rows, results_dir, goal)
        _mirror_to_run(row, results_dir)
        if row["status"] == "ok" and math.isfinite(row["metric"]):
            # Directly update the Optuna DB: mark the FAIL'd trial as COMPLETE and record metric.
            import sqlite3
            db_path = os.path.join(results_dir, "study.db")
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("UPDATE trials SET state='COMPLETE', datetime_complete=datetime('now') WHERE number=? AND state='FAIL'",
                      (ft.number,))
            # insert the trial value
            c.execute("SELECT trial_id FROM trials WHERE number=?", (ft.number,))
            trial_id = c.fetchone()[0]
            c.execute("INSERT OR REPLACE INTO trial_values (trial_id, objective, value, value_type) VALUES (?, 0, ?, 'FINITE')",
                      (trial_id, row["metric"]))
            conn.commit()
            conn.close()
            print(f"[sweep] trial {ft.number} resumed successfully: metric={row['metric']}")
            # Reload the study to pick up the DB change
            study = optuna.load_study(
                study_name=os.path.basename(results_dir),
                storage=f"sqlite:///{os.path.join(results_dir, 'study.db')}",
            )
        else:
            print(f"[sweep] trial {ft.number} retry failed: status={row['status']}")

    # ------------------------------------------------------------------
    # Phase 1: run remaining NEW trials
    # ------------------------------------------------------------------
    # num_samples = TOTAL trial budget (not per-invocation): on resume, only run what's left,
    # counting trials already finished in the store (so a resumed study tops up to num_samples).
    n_target = int(cfg.get("num_samples", 20))
    done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED", "FAIL")])
    remaining = max(0, n_target - done)
    print(f"[sweep] optuna: {done} finished trial(s) in store, running {remaining} more (target {n_target}).")
    try:
        study.optimize(objective, n_trials=remaining)
    except KeyboardInterrupt:
        print("\n[sweep] interrupted — partial results saved.")
    except RuntimeError as e:
        print(f"\n[sweep] ABORTED: {e}")
    if study.best_trial is not None:
        print(f"\n[sweep] optuna best value={study.best_value}")
        print(f"[sweep] optuna best params={study.best_params}")
    print(f"[sweep] resume/inspect: optuna-dashboard sqlite:///{os.path.join(results_dir, 'study.db')}")


def _write_results(rows: list[dict], results_dir: str, goal: str) -> dict | None:
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "results.csv"), index=False)
    ok = df[df["status"] == "ok"].dropna(subset=["metric"])
    if ok.empty:
        return None
    ranked = ok.sort_values("metric", ascending=(goal == "min"))
    ranked.to_csv(os.path.join(results_dir, "results_ranked.csv"), index=False)
    print("\n[sweep] top trials:")
    print(ranked.head(10).to_string(index=False))
    return ranked.iloc[0].to_dict()


if __name__ == "__main__":
    main()
