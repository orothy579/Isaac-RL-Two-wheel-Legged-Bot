"""Fake the outputs of one h200_blind15_queue.sh job (DRY_RUN=1): checkpoints of a training run, the distilled
student, or an evaluation JSON with random scores. Lets the queue's control flow be tested without the GPU.

    python3 queue_dry_run.py <run root G> <script> <script args...>
"""

import json
import os
import random
import sys
import time

G, script, args = sys.argv[1], sys.argv[2], sys.argv[3:]
opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
if script.endswith("train.py"):
    # train.py resumed from model_N runs N .. N + iterations - 1 (the Gain resume: model_500 + 2250 -> model_2749)
    start = int(opt["--checkpoint"].split("_")[1].split(".")[0]) if "--resume" in opt else 0
    last = start + int(opt["--max_iterations"]) - 1
    run = os.path.join(G, time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + opt["--run_name"])
    os.makedirs(run, exist_ok=True)
    for it in sorted({*range(0, last + 1, 100), last}):
        open(os.path.join(run, f"model_{it}.pt"), "w").close()
    time.sleep(1.1)  # distinct timestamps
elif script.endswith("distill_stair_step.py"):
    os.makedirs(opt["--out_dir"], exist_ok=True)
    open(os.path.join(opt["--out_dir"], "student_1499.pt"), "w").close()
elif script.endswith("eval_stair_step.py"):
    levels = {k: {"riser_height_m": h, "envs_climbed_1plus": random.random(), "risers_climbed_mean": 5 * random.random()}
              for k, h in (("0", 0.0933), ("3", 0.1133), ("6", 0.1333), ("9", 0.15))}
    with open(opt["--out"], "w") as f:
        json.dump({"checkpoint": opt["--checkpoint"], "levels": levels}, f)
