#!/usr/bin/env python3
"""Run Gain and blind-student fine-tuning on distinct GPUs, or sequentially on one."""
import argparse
import csv
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT.parent
STUDENT_CHECKPOINT = Path(
    "logs/co_rl/Flamingo_Light_Rough_Stair_Step_Distill/"
    "2026-10-29_09-52-52_gain2999_h50/student_1499.pt"
)


def discover_gpus():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=30)
    return [{"index": int(row[0].strip()), "uuid": row[1].strip(), "name": row[2].strip()}
            for row in csv.reader(result.stdout.splitlines()) if row]


def plan_jobs(gpus, gain_checkpoint, student_checkpoint):
    if not gpus:
        raise RuntimeError("No GPUs available for H200 jobs")
    return [
        {"job": "gain", "gpu": gpus[0]["index"], "checkpoint": str(gain_checkpoint), "iterations": 5000},
        {"job": "student", "gpu": gpus[min(1, len(gpus) - 1)]["index"],
         "checkpoint": str(student_checkpoint), "iterations": 3000},
    ]


def execute_jobs(plan, log_dir, report, worker=None):
    worker = worker or Path(__file__).with_name("scale_train.py")
    mode = "parallel" if len({job["gpu"] for job in plan}) == len(plan) else "sequential"
    results = {}
    report(phase="running", mode=mode, jobs=plan, results=results.copy())
    with ExitStack() as stack:
        def launch(job):
            log = stack.enter_context((log_dir / f"{job['job']}_worker.log").open("w"))
            command = [sys.executable, "-u", str(worker), "--job", job["job"],
                       "--gpu", str(job["gpu"]), "--checkpoint", job["checkpoint"],
                       "--iterations", str(job["iterations"])]
            return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        if mode == "parallel":
            processes = [(job, launch(job)) for job in plan]
            while processes:
                for job, process in processes[:]:
                    code = process.poll()
                    if code is not None:
                        results[job["job"]] = code
                        processes.remove((job, process))
                        report(phase="running", mode=mode, jobs=plan, results=results.copy())
                if processes:
                    time.sleep(1)
        else:
            for job in plan:
                # A failed Gain job is recorded; the independent student still gets its turn.
                process = launch(job)
                report(phase="running", mode=mode, jobs=plan, active_job=job["job"], results=results.copy())
                results[job["job"]] = process.wait()
        report(phase="complete" if all(code == 0 for code in results.values()) else "failed",
               mode=mode, jobs=plan, results=results.copy())
    return 0 if all(code == 0 for code in results.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gain-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, default=STUDENT_CHECKPOINT)
    args = parser.parse_args()
    lock = (BASE / "dispatch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for checkpoint in [args.gain_checkpoint, args.student_checkpoint]:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    gpus = discover_gpus()
    plan = plan_jobs(gpus, args.gain_checkpoint, args.student_checkpoint)

    def report(**data):
        data.update(updated_unix=time.time(), gpus=gpus)
        temp = BASE / "job_queue_status.tmp"
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(BASE / "job_queue_status.json")
        print(json.dumps(data), flush=True)

    return execute_jobs(plan, BASE, report)


if __name__ == "__main__":
    sys.exit(main())
