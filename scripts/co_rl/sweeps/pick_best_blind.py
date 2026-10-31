"""Pick the checkpoint with the best 15 cm result from Isaac and MuJoCo evaluation JSON files (no simulator needed).

Isaac files come from ``scripts/co_rl/eval_stair_step.py``, MuJoCo files from ``scripts/co_rl/cosim_eval_policy.py``
(``"sim": "mujoco"``). The goal is >= 50 % of episodes climbing 3 consecutive risers on row 9 (15 cm) in BOTH
simulators, so a checkpoint scores the mean over the simulators of its 3-riser share on ``--level`` (several files
of one simulator, e.g. the nominal and the robust Isaac play task, are averaged first; a simulator evaluated for
some checkpoint but not for this one counts 0), ties broken the same way by the mean number of risers. Files
written before the 3-riser share existed are skipped. Prints the best checkpoint path, or with ``--table`` every
checkpoint with, per row, the >= 1 / >= 3 riser shares and the mean number of risers.

    python3 scripts/co_rl/sweeps/pick_best_blind.py a.json b.json ...            # -> best model_N.pt
    python3 scripts/co_rl/sweeps/pick_best_blind.py --table a.json b.json ...
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("evals", nargs="*")
ap.add_argument("--level", default="9")
ap.add_argument("--table", action="store_true")
a = ap.parse_args()

by_ckpt: dict[str, dict[str, list[dict]]] = {}
sims: set[str] = set()
for path in a.evals:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        rep = json.load(f)
    row = rep.get("levels", {}).get(a.level)
    if row is None or "envs_climbed_3plus" not in row or not rep.get("checkpoint"):
        print(f"[pick_best] skipped {path}: no 3-riser share at row {a.level}", file=sys.stderr)
        continue
    sim = rep.get("sim", "isaac")
    sims.add(sim)
    by_ckpt.setdefault(rep["checkpoint"], {}).setdefault(sim, []).append({"file": path, **row, "all": rep["levels"]})
if not by_ckpt:
    sys.exit("no usable evaluation")


def score(per_sim: dict[str, list[dict]]) -> tuple[float, float]:
    def avg(key: str, sim: str) -> float:
        rows = per_sim.get(sim, [])
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return (sum(avg("envs_climbed_3plus", s) for s in sims) / len(sims),
            sum(avg("risers_climbed_mean", s) for s in sims) / len(sims))


ranked = sorted(by_ckpt.items(), key=lambda kv: score(kv[1]), reverse=True)
if a.table:
    print(f"per row: climbed >= 1 riser / >= 3 consecutive risers / mean risers   (score: mean over {sorted(sims)})")
    for ckpt, per_sim in ranked:
        s3, sm = score(per_sim)
        print(f"{ckpt}  score {s3 * 100:.1f} % / {sm:.2f}")
        for sim, rows in sorted(per_sim.items()):
            for r in rows:
                cells = "  ".join(
                    f"{float(v['riser_height_m']) * 100:.1f}cm {v['envs_climbed_1plus'] * 100:5.1f}%/"
                    f"{v.get('envs_climbed_3plus', float('nan')) * 100:5.1f}%/{v['risers_climbed_mean']:.2f}"
                    for v in r["all"].values())
                print(f"    {sim:6s} [{os.path.basename(r['file'])}]  {cells}")
else:
    print(ranked[0][0])
