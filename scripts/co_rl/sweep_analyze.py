# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Analyze an Optuna sweep (Phase-1 step 2-3): rank parameters by correlation with the metric
and by Optuna importance, print the Top-K, and optionally emit a *focused* sweep YAML that
narrows the Top-K around the best trial and fixes the rest.

Usage::

    # analyze a finished/partial sweep
    python scripts/co_rl/sweep_analyze.py --results_dir logs/co_rl/<exp>/<algo>/_sweeps/<name>_<stamp>
    python scripts/co_rl/sweep_analyze.py --study <path/to/study.db>

    # analyze AND generate a focused sweep yaml from the Top-3 (needs the broad yaml as base)
    python scripts/co_rl/sweep_analyze.py --results_dir <dir> \
        --make_focused scripts/co_rl/sweeps/stair_jump_focused_auto.yaml \
        --base scripts/co_rl/sweeps/stair_jump_example.yaml --top_k 3

Correlation is Spearman (rank-based, numpy only). Importance is Optuna's default evaluator
(falls back to |correlation| if it can't run — e.g. too few trials).
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def load_study(study_path: str | None, results_dir: str | None):
    import optuna

    if not study_path and results_dir:
        study_path = os.path.join(results_dir, "study.db")
    if not study_path or not os.path.isfile(study_path):
        raise FileNotFoundError(f"study.db not found (study={study_path}, results_dir={results_dir})")
    storage = f"sqlite:///{os.path.abspath(study_path)}"
    names = optuna.study.get_all_study_names(storage)
    return optuna.load_study(study_name=names[0], storage=storage)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-based correlation, numpy only (nan if either side is constant)."""
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze(study):
    completed = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    n = len(completed)
    print(f"[analyze] completed trials: {n}  (need ~15-20 for reliable importance)")
    if n < 2:
        raise SystemExit("[analyze] not enough completed trials to analyze.")

    names = sorted({k for t in completed for k in t.params})
    y = np.array([t.value for t in completed], dtype=float)

    corr = {}
    for p in names:
        col = np.array([t.params.get(p, np.nan) for t in completed], dtype=float)
        corr[p] = float("nan") if np.isnan(col).any() or np.nanstd(col) == 0 else spearman(col, y)

    imp = {}
    try:
        import optuna

        try:  # PedAnova needs no sklearn; fall back to the default (sklearn) evaluator
            ev = optuna.importance.PedAnovaImportanceEvaluator()
            imp = dict(optuna.importance.get_param_importances(study, evaluator=ev))
        except Exception:  # noqa: BLE001
            imp = dict(optuna.importance.get_param_importances(study))
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] Optuna importance unavailable ({type(e).__name__}: {e}); ranking by |corr|.")

    # rank: prefer Optuna importance, else |Spearman|
    def key(p):
        return imp.get(p, abs(corr[p]) if not np.isnan(corr[p]) else -1.0)

    ranked = sorted(names, key=key, reverse=True)
    print(f"\n{'param':<52} {'importance':>11} {'spearman':>9}")
    for p in ranked:
        im = f"{imp[p]:.3f}" if p in imp else "-"
        cr = f"{corr[p]:+.3f}" if not np.isnan(corr[p]) else "const"
        print(f"{p:<52} {im:>11} {cr:>9}")
    return ranked, corr, imp


def make_focused(base_yaml: str, study, top: list[str], out_path: str, span_frac: float = 0.25):
    import yaml

    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)
    best = study.best_trial
    print(f"\n[focused] best trial #{best.number} value={best.value:.3f}")
    newp = {}
    for p, spec in cfg["parameters"].items():
        bv = best.params.get(p)
        if p in top and "min" in spec:  # important continuous param -> narrow window around best
            lo, hi = float(spec["min"]), float(spec["max"])
            span = (hi - lo) * span_frac
            entry = {"min": round(max(lo, bv - span), 6), "max": round(min(hi, bv + span), 6)}
            if spec.get("log"):
                entry["log"] = True
            newp[p] = entry
        elif p in top:  # important categorical -> keep its choices
            newp[p] = spec
        else:  # unimportant -> fix to the best trial's value
            newp[p] = {"values": [bv]}
    cfg["parameters"] = newp
    cfg["seed"] = int(cfg.get("seed", 42)) + 1  # fresh study
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[focused] wrote {out_path}  (Top-{len(top)} narrowed: {top}; rest fixed to best)")


def main():
    p = argparse.ArgumentParser(description="Analyze an Optuna sweep and (optionally) emit a focused yaml.")
    p.add_argument("--study", default=None, help="Path to study.db")
    p.add_argument("--results_dir", default=None, help="Sweep results dir (contains study.db)")
    p.add_argument("--top_k", type=int, default=3, help="Number of top parameters to focus on.")
    p.add_argument("--make_focused", default=None, help="Output path for a generated focused sweep yaml.")
    p.add_argument("--base", default=None, help="Base (broad) yaml to derive the focused yaml from.")
    args = p.parse_args()

    study = load_study(args.study, args.results_dir)
    ranked, corr, imp = analyze(study)
    top = ranked[: args.top_k]
    print(f"\n[analyze] Top-{args.top_k} parameters: {top}")

    if args.make_focused:
        if not args.base:
            raise SystemExit("--make_focused needs --base <broad yaml>")
        make_focused(args.base, study, top, args.make_focused)


if __name__ == "__main__":
    main()
