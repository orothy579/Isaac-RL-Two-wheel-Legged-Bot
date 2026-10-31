"""Pick the checkpoint with the best 15 cm result from ``eval_stair_step.py`` JSON files (no Isaac needed).

Score per evaluation: share of envs that climbed >= 1 riser on ``--level`` (row 9 = 15 cm), ties broken by the
mean number of risers. Several files of the same checkpoint (e.g. the nominal and the robust play task) are
averaged. Prints the best checkpoint path, or with ``--table`` every checkpoint with its numbers.

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

by_ckpt: dict[str, list[dict]] = {}
for path in a.evals:
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        rep = json.load(f)
    if a.level not in rep.get("levels", {}):
        continue
    by_ckpt.setdefault(rep["checkpoint"], []).append({"file": path, **rep["levels"][a.level],
                                                      "all": rep["levels"]})
if not by_ckpt:
    sys.exit("no usable evaluation")


def score(rows: list[dict]) -> tuple[float, float]:
    return (sum(r["envs_climbed_1plus"] for r in rows) / len(rows), sum(r["risers_climbed_mean"] for r in rows) / len(rows))


ranked = sorted(by_ckpt.items(), key=lambda kv: score(kv[1]), reverse=True)
if a.table:
    for ckpt, rows in ranked:
        for r in rows:
            cells = "  ".join(f"{float(v['riser_height_m']) * 100:.1f}cm {v['envs_climbed_1plus'] * 100:5.1f}%/{v['risers_climbed_mean']:.2f}"
                              for v in r["all"].values())
            print(f"{ckpt}  [{os.path.basename(r['file'])}]  {cells}")
else:
    print(ranked[0][0])
