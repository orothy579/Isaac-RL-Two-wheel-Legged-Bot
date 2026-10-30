#!/usr/bin/env python3
"""Smoke-test real Flamingo physics/PPO, benchmark up to 65536, then train.

H200 is outside Isaac Sim 5.1's officially supported RTX hardware. An actual
completed task rollout is required; CUDA availability alone is insufficient.
This is a scaled baseline, not a reproduction of ACP or OmniReset.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
from functools import partial

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT.parent
PROFILES = {
    "gain": {
        "task": "Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-v1-ppo",
        "policy_stacks": 2, "iterations": 5000, "widen": True, "run_name": "h200_scaled",
    },
    "student": {
        "task": "Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo",
        "policy_stacks": 49, "iterations": 3000, "widen": False, "run_name": "student_ft",
    },
}
RUN_ROOT = ROOT / "logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo"


def overrides(n, gpu=0):
    factor = math.ceil(n / 4096)
    return {
        # train.py does not copy AppLauncher --device into either of these configs.
        "env.sim.device": f"cuda:{gpu}",
        "agent.device": f"cuda:{gpu}",
        "agent.algorithm.num_mini_batches": max(4, n // 2048),
        "env.sim.physx.gpu_max_rigid_patch_count": max(5 * 2**15, n * 32),
        "env.sim.physx.gpu_found_lost_pairs_capacity": max(2**21, n * 64),
        "env.sim.physx.gpu_heap_capacity": 2**26 * min(factor, 8),
        "env.sim.physx.gpu_collision_stack_size": 2**26 * min(factor, 4),
    }


def write_status(profile="gain", **data):
    data["updated_unix"] = time.time()
    data["job"] = profile
    filename = "training_status" if profile == "gain" else "student_training_status"
    temp = BASE / f"{filename}.tmp"
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(BASE / f"{filename}.json")
    print(json.dumps(data), flush=True)


def complete_run(name, iterations):
    markers = list(RUN_ROOT.glob(f"*_{name}/batch_complete.json"))
    if len(markers) != 1:
        raise RuntimeError(f"Expected one completed run for {name}, got {len(markers)}")
    marker = markers[0]
    data = json.loads(marker.read_text())
    if data["iterations_requested"] != iterations or data["last_iteration"] < iterations - 1:
        raise RuntimeError(f"Incomplete training marker: {marker}")
    ckpt = marker.parent / data["checkpoint"]
    if not ckpt.is_file():
        raise RuntimeError(f"Checkpoint missing: {ckpt}")
    # Inspect weights and the optimizer for non-finite values, not just log strings.
    import torch
    def check(obj):
        if isinstance(obj, torch.Tensor) and not torch.isfinite(obj).all():
            raise RuntimeError(f"Nonfinite checkpoint: {ckpt}")
        if isinstance(obj, dict):
            for value in obj.values():
                check(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                check(value)
    check(torch.load(ckpt, map_location="cpu", weights_only=False))
    return ckpt


def training_command(n, iterations, checkpoint, name, spec, profile="gain", gpu=0, kit_args=""):
    cfg = PROFILES[profile]
    command = ["bash", "scripts/co_rl/h200/run.sh", "-u", "scripts/co_rl/train.py",
               "--task", cfg["task"], "--algo", "ppo", "--headless", "--num_envs", str(n),
               "--device", f"cuda:{gpu}", "--num_policy_stacks", str(cfg["policy_stacks"]),
               "--num_critic_stacks", "2",
               "--max_iterations", str(iterations), "--batch_exit", "--run_name", name,
               "--param_overrides", str(spec), "--warmstart_ckpt", str(checkpoint),
               "--warmstart_std_scale", "1.0"]
    if cfg["widen"]:
        command += ["--warmstart_widen"]
    # The headless experience otherwise enables the renderer's multi-GPU mode.
    flags = "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
    command += ["--kit_args", (flags + " " + kit_args).strip()]
    return command


def run(n, iterations, checkpoint, name, kit_args="", timeout=3600, profile="gain", gpu=0):
    spec = BASE / f"{name}_overrides.json"
    spec.write_text(json.dumps(overrides(n, gpu), indent=2))
    log = BASE / f"{name}.log"
    command = training_command(n, iterations, checkpoint, name, spec, profile, gpu, kit_args)
    started = time.monotonic()
    measurements = []
    with log.open("w") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise RuntimeError(f"Timed out: {name}; see {log}")
                sample = subprocess.run(["nvidia-smi", "-i", str(gpu),
                                         "--query-gpu=memory.used,memory.total,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if sample.returncode == 0:
                    fields = [x.strip() for x in sample.stdout.splitlines()[0].split(",")]
                    used, total = int(fields[0]), int(fields[1])
                    util = int(fields[2]) if fields[2].isdigit() else None
                    measurements.append({"memory_mib": used, "util_percent": util})
                    if used / total > 0.96:
                        raise RuntimeError(f"GPU memory safety margin exceeded at {n} environments")
                cgroup_limit = Path("/sys/fs/cgroup/memory.max")
                cgroup_used = Path("/sys/fs/cgroup/memory.current")
                if cgroup_limit.exists() and cgroup_used.exists():
                    limit = cgroup_limit.read_text().strip()
                    if limit.isdigit() and int(cgroup_used.read_text()) / int(limit) > 0.94:
                        raise RuntimeError(f"Container memory safety margin exceeded at {n} environments")
                time.sleep(5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    if process.returncode:
        raise RuntimeError(f"{name} exited {process.returncode}; see {log}")
    ckpt = complete_run(name, iterations)
    text = log.read_text(errors="replace")
    # The term name nonfinite_state is normal; positive termination counts are not.
    if re.search(r"Episode_Termination/nonfinite_state:\s*(?!0\.0000)([0-9]+\.[0-9]+)", text):
        raise RuntimeError(f"Nonfinite simulator state was observed; see {log}")
    speeds = [float(x) for x in re.findall(r"Computation:\s*([0-9.]+) steps/s", text)]
    if not speeds:
        raise RuntimeError(f"No measured PPO throughput in {log}")
    result = {"job": profile, "gpu": gpu, "num_envs": n,
              "steps_per_second": statistics.median(speeds[-10:]),
              "peak_gpu_memory_mib": max((x["memory_mib"] for x in measurements), default=None),
              "log": str(log), "checkpoint": str(ckpt)}
    (BASE / f"{name}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--job", choices=PROFILES, default="gain")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    cfg = PROFILES[args.job]
    args.iterations = args.iterations if args.iterations is not None else cfg["iterations"]
    # Separate profiles may run concurrently, but never share a physical GPU.
    locks = [(BASE / f"training_{args.job}.lock").open("w"),
             (BASE / f"gpu_{args.gpu}.lock").open("w")]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = str(int(time.time()))
    status = partial(write_status, profile=args.job, gpu=args.gpu)
    execute = partial(run, profile=args.job, gpu=args.gpu)
    prefix = "h200" if args.job == "gain" else "h200_student"
    try:
        if not args.checkpoint.is_file():
            raise RuntimeError(f"Starting checkpoint missing: {args.checkpoint}")
        if args.job == "student":
            # Reject a teacher/wrong-history checkpoint before spending GPU time.
            import torch
            state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
            if state["actor.0.weight"].shape[1] != 804 or state["critic.0.weight"].shape[1] != 216:
                raise RuntimeError("Blind student checkpoint must have actor input 804 and critic input 216")
        kit = ""
        status(phase="smoke_test", official_h200_support=False)
        try:
            execute(64, 3, args.checkpoint, f"{prefix}_smoke_{stamp}", timeout=1200)
        except RuntimeError as first:
            status(phase="smoke_test_no_renderer", first_attempt_error=str(first))
            kit = "--/renderer/enabled=false --/app/renderer/enabled=false --/app/window/enabled=false"
            execute(64, 3, args.checkpoint, f"{prefix}_smoke_norender_{stamp}", kit_args=kit, timeout=1200)
        passed = []
        for n in [4096, 8192, 16384, 32768, 65536]:
            status(phase="benchmarking", num_envs=n, passed=passed)
            try:
                passed.append(execute(n, 30, args.checkpoint, f"{prefix}_bench{n}_{stamp}", kit_args=kit))
            except RuntimeError as exc:
                # Stop on a software/physics fault. Only capacity failures can use a smaller size.
                log = BASE / f"{prefix}_bench{n}_{stamp}.log"
                contents = log.read_text(errors="replace") if log.exists() else ""
                if passed and re.search(r"out of memory|capacity|buffer overflow|memory safety margin", contents + str(exc), re.I):
                    status(phase="capacity_limit", num_envs=n, error=str(exc))
                    break
                raise
        best = max(passed, key=lambda r: r["steps_per_second"])
        # Prefer the largest size if its throughput is within 10% of the best.
        selected = max((r for r in passed if r["steps_per_second"] >= 0.9 * best["steps_per_second"]),
                       key=lambda r: r["num_envs"])
        benchmark_file = "benchmark_results.json" if args.job == "gain" else "student_benchmark_results.json"
        (BASE / benchmark_file).write_text(json.dumps({"passed": passed, "selected": selected}, indent=2))
        status(phase="training", selected=selected, iterations=args.iterations,
                     method="scaled existing PPO baseline; motion priors/diverse reset bank not yet implemented")
        result = execute(selected["num_envs"], args.iterations, args.checkpoint,
                         f"{cfg['run_name']}_{stamp}", kit_args=kit, timeout=7 * 86400)
        status(phase="complete", result=result)
    except Exception as exc:
        status(phase="failed", error=str(exc),
                     note="Inspect logs; no successful H200 training is claimed")
        raise


if __name__ == "__main__":
    main()
