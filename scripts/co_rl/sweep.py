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
    try:
        with open(logfile) as f:
            for line in f:
                if LOG_DIR_MARKER in line:
                    return line.split(LOG_DIR_MARKER, 1)[1].strip()
    except OSError:
        return None
    return None


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


def build_cmd(python: str, task: str, algo: str, max_iter: int, run_name: str,
              override_path: str, base_args: list[str]) -> list[str]:
    return [
        python, TRAIN_PY,
        "--task", task, "--algo", algo,
        "--max_iterations", str(max_iter),
        "--run_name", run_name,
        "--param_overrides", override_path,
        *[str(a) for a in base_args],
    ]


def run_trial(idx: int, params: dict, cfg: dict, results_dir: str, python: str) -> dict:
    run_name = f"sweep{idx:03d}"
    override_path = os.path.join(results_dir, f"{run_name}_overrides.json")
    with open(override_path, "w") as f:
        json.dump(params, f, indent=2)
    logfile = os.path.join(results_dir, f"{run_name}.log")

    cmd = build_cmd(python, cfg["task"], cfg.get("algo", "ppo"), int(cfg["max_iterations"]),
                    run_name, override_path, cfg.get("base_args", []))
    print(f"[sweep] trial {idx}: {params}")
    print(f"[sweep]   $ {' '.join(cmd)}")
    with open(logfile, "w") as lf:
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


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="CO-RL reward/hyperparameter sweep driver.")
    p.add_argument("--config", required=True, help="Path to the sweep YAML config.")
    p.add_argument("--dry_run", action="store_true", help="List the trials and exit (no training).")
    p.add_argument("--python", default=sys.executable, help="Python interpreter for train.py subprocesses.")
    p.add_argument("--results_dir", default=None, help="Where to write trial logs/overrides/results.")
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
        _run_optuna(cfg, results_dir, args.python, goal)
        return
    if method not in ("grid", "random"):
        raise ValueError(f"method must be 'grid', 'random', or 'optuna', got '{method}'")

    rng = random.Random(cfg.get("seed", 42))
    trials = expand_grid(cfg["parameters"]) if method == "grid" \
        else sample_random(cfg["parameters"], int(cfg.get("num_samples", 10)), rng)
    rows = []
    try:
        for i, t in enumerate(trials):
            rows.append(run_trial(i, t, cfg, results_dir, args.python))
            _write_results(rows, results_dir, goal)  # checkpoint after every trial
    except KeyboardInterrupt:
        print("\n[sweep] interrupted — writing partial results.")

    best = _write_results(rows, results_dir, goal)
    print(f"\n[sweep] done. best trial: {best}")


def _run_optuna(cfg: dict, results_dir: str, python: str, goal: str) -> None:
    """TPE-sampled Optuna study over the same parameter spec; resumable via a SQLite store.

    Reuses ``run_trial`` (subprocess train.py + ``--param_overrides``) and ``extract_metric``.
    A failed run (non-zero exit / missing metric) is reported as pruned so the study continues.
    """
    import optuna

    rows: list[dict] = []

    def objective(trial):
        params = suggest_params(trial, cfg["parameters"])
        row = run_trial(trial.number, params, cfg, results_dir, python)
        rows.append(row)
        _write_results(rows, results_dir, goal)  # checkpoint after every trial
        if row["status"] != "ok" or not math.isfinite(row["metric"]):
            raise optuna.TrialPruned()
        return row["metric"]

    study = optuna.create_study(
        study_name=os.path.basename(results_dir),
        direction="maximize" if goal == "max" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
        storage=f"sqlite:///{os.path.join(results_dir, 'study.db')}",
        load_if_exists=True,
    )
    try:
        study.optimize(objective, n_trials=int(cfg.get("num_samples", 20)))
    except KeyboardInterrupt:
        print("\n[sweep] interrupted — partial results saved.")
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
