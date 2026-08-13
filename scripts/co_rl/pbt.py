# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Population-Based Training driver for CO-RL (research_advice.md Phase 2 / Approach B).

Evolves a small population of reward/balancer configurations ACROSS generations while the
policies keep training. Unlike Optuna (independent full runs), PBT shares progress: each
generation every member trains ``iters_per_gen`` from its own checkpoint; then the bottom
member(s) copy the checkpoint AND params of a top member (exploit) and perturb the params
(explore). Single-GPU sequential.

Usage::

    python scripts/co_rl/pbt.py --config scripts/co_rl/sweeps/pbt_stair_jump.yaml
    # resume a crashed run:
    python scripts/co_rl/pbt.py --config <same.yaml> --results_dir <existing _pbt/... dir>

Caveats (by design):
* Each generation re-creates the env, so the terrain curriculum restarts at level 0 and the
  metric (final_mean of terrain) measures how fast/high the member RE-climbs within one
  generation — favors robust, quickly-recovering policies.
* Checkpoint transfer uses train.py's ``--warmstart_ckpt`` (load_transfer): actor/critic
  weights are copied, optimizer/iteration reset each generation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

import sweep  # reuse: REPO_ROOT, parse_log_dir, extract_metric, _latest_checkpoint, build_cmd


def run_member(gen: int, idx: int, params: dict, ckpt: str, cfg: dict, results_dir: str,
               python: str) -> dict:
    """Train one member for one generation; return {log_dir, metric, ckpt, status}."""
    run_name = f"pbt_g{gen}m{idx}"
    override_path = os.path.join(results_dir, f"{run_name}_overrides.json")
    with open(override_path, "w") as f:
        json.dump(params, f, indent=2)
    logfile = os.path.join(results_dir, f"{run_name}.log")

    extra = list(cfg.get("base_args", [])) + ["--warmstart_ckpt", ckpt]
    cmd = sweep.build_cmd(python, cfg["task"], cfg.get("algo", "ppo"),
                          int(cfg["iters_per_gen"]), run_name, override_path, extra)
    print(f"[pbt] g{gen} m{idx}: {params}")
    print(f"[pbt]   $ {' '.join(cmd)}", flush=True)
    with open(logfile, "w") as lf:
        proc = subprocess.run(cmd, cwd=sweep.REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT,
                              env=os.environ)

    out = {"log_dir": None, "metric": float("nan"), "ckpt": None, "status": "ok"}
    if proc.returncode != 0:
        out["status"] = f"train_exit_{proc.returncode}"
        return out
    log_dir = sweep.parse_log_dir(logfile)
    out["log_dir"] = log_dir
    if not log_dir or not os.path.isdir(log_dir):
        out["status"] = "no_log_dir"
        return out
    try:
        out["metric"] = sweep.extract_metric(log_dir, cfg.get("metric", {}))
    except Exception as e:  # noqa: BLE001
        out["status"] = f"metric_error: {e}"
    fname = sweep._latest_checkpoint(log_dir)  # returns the FILENAME only
    out["ckpt"] = os.path.join(os.path.abspath(log_dir), fname) if fname else None
    if out["ckpt"] is None:
        out["status"] = "no_checkpoint"
    return out


def perturb(params: dict, evolve: dict, rng: random.Random, factor: float) -> dict:
    """Multiply each evolved param by U{1/factor, factor}, clipped to its bounds."""
    new = dict(params)
    for key, bounds in evolve.items():
        f = factor if rng.random() < 0.5 else 1.0 / factor
        new[key] = float(min(max(new[key] * f, bounds["min"]), bounds["max"]))
    return new


def main():
    p = argparse.ArgumentParser(description="Population-Based Training driver for CO-RL.")
    p.add_argument("--config", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--results_dir", default=None, help="Existing _pbt dir to resume.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rng = random.Random(cfg.get("seed", 0))
    n_pop = int(cfg["population"])
    n_gen = int(cfg["generations"])
    n_exploit = max(1, int(n_pop * float(cfg.get("exploit_frac", 0.25))))
    factor = float(cfg.get("perturb_factor", 1.2))
    evolve = cfg["evolve"]  # {dotpath: {min,max}}

    if args.results_dir:
        results_dir = os.path.abspath(args.results_dir)
        with open(os.path.join(results_dir, "pbt_state.json")) as f:
            state = json.load(f)
        print(f"[pbt] resuming from {results_dir} at generation {state['generation']}")
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = os.path.join(sweep.REPO_ROOT, "logs", "co_rl", cfg["experiment_name"],
                                   cfg.get("algo", "ppo"), "_pbt", f"pbt_{stamp}")
        os.makedirs(results_dir, exist_ok=True)
        center = dict(cfg["center_params"])
        members = []
        for i in range(n_pop):
            params = dict(center) if i == 0 else perturb(center, evolve, rng, factor)
            members.append({"params": params, "ckpt": cfg["init_ckpt"], "metric": None, "done": False})
        state = {"generation": 0, "members": members}
        with open(os.path.join(results_dir, "pbt_state.json"), "w") as f:
            json.dump(state, f, indent=2)
    print(f"[pbt] results dir: {results_dir}")

    hist_path = os.path.join(results_dir, "history.csv")
    new_hist = not os.path.exists(hist_path)
    hist = open(hist_path, "a", newline="")
    writer = csv.writer(hist)
    if new_hist:
        writer.writerow(["generation", "member", "metric", "status", "ckpt", "log_dir", "params"])

    state_path = os.path.join(results_dir, "pbt_state.json")

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    for gen in range(state["generation"], n_gen):
        # -- train every member for one generation (skip members already done THIS
        # generation, e.g. after a crash mid-generation resumed via --results_dir).
        for i, m in enumerate(state["members"]):
            if m.get("done"):
                print(f"[pbt] g{gen} m{i}: already done this generation "
                      f"(metric={m['metric']:.3f}) — skipping", flush=True)
                continue
            res = run_member(gen, i, m["params"], m["ckpt"], cfg, results_dir, args.python)
            if res["status"].startswith("train_exit") or res["status"] == "no_log_dir":
                print(f"[pbt] ABORTED at g{gen} m{i}: training didn't start "
                      f"(status={res['status']}). Fix the env, then resume with "
                      f"--results_dir {results_dir}")
                hist.close()
                sys.exit(1)  # non-zero so a chain script reports the failure
            if res["ckpt"]:  # keep training this member from its new checkpoint next gen
                m["ckpt"] = res["ckpt"]
            m["metric"] = res["metric"]
            m["done"] = True
            save_state()  # persist after EVERY member so a crash loses at most one run
            writer.writerow([gen, i, res["metric"], res["status"], res["ckpt"],
                             res["log_dir"], json.dumps(m["params"])])
            hist.flush()
            print(f"[pbt] g{gen} m{i}: metric={res['metric']:.3f} status={res['status']}", flush=True)

        # -- exploit + explore (only if there IS a next generation to run it for; on the
        # last generation this would just mutate the final population pointlessly and
        # risks overwriting a top member if it happens to rank in the bottom slice of a
        # later spurious pass).
        if gen < n_gen - 1:
            ranked = sorted(range(n_pop), key=lambda i: state["members"][i]["metric"], reverse=True)
            top, bottom = ranked[:n_exploit], ranked[-n_exploit:]
            for b in bottom:
                src = state["members"][rng.choice(top)]
                print(f"[pbt] g{gen}: member {b} exploits {ranked[:n_exploit]} "
                      f"(copies ckpt+params of a top member, then perturbs)")
                state["members"][b] = {
                    "params": perturb(src["params"], evolve, rng, factor),
                    "ckpt": src["ckpt"],
                    "metric": None,
                    "done": False,
                }
            for m in state["members"]:  # every member (survivors incl.) retrains next gen
                m["done"] = False
        state["generation"] = gen + 1
        save_state()

    hist.close()
    best = max(state["members"], key=lambda m: (m["metric"] if m["metric"] is not None else -1e9))
    print(f"\n[pbt] DONE. best member: metric={best['metric']:.3f}")
    print(f"[pbt]   params={json.dumps(best['params'], indent=2)}")
    print(f"[pbt]   ckpt={best['ckpt']}")
    print(f"[pbt] full history: {hist_path}")


if __name__ == "__main__":
    main()
