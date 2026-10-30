#!/usr/bin/env python3
"""Wait for the complete local two-stage run, then copy and launch on H200.

Uses an already authenticated SSH control connection; never stores a password.
All remote writes happen after the completion barrier.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime, timezone
from dispatch import STUDENT_CHECKPOINT

ROOT = Path(__file__).resolve().parents[3]
REMOTE_BASE = "/home/work/postech/flamingo_light"
REMOTE_REPO = REMOTE_BASE + "/Isaac-RL-Two-wheel-Legged-Bot"
FINAL_ROOT = ROOT / "logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo"
SSH = ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
       "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
       "-o", "ControlPath=/tmp/flamingo-h200-ssh-%C", "-p", "10537",
       "work@proxy2.ainexus.ktcloud.com"]


def write_status(path, phase, **kwargs):
    data = {"phase": phase, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "queued_jobs": ["gain", "student"],
            "gpu_scheduling": "different GPUs in parallel; Gain then student on one GPU", **kwargs}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)
    print(json.dumps(data), flush=True)


def remote(command, **kwargs):
    return subprocess.run(SSH + [command], check=True, **kwargs)


def proc_identity(pid):
    try:
        # Fields following the process name begin at stat field 3.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def training_processes():
    """Look at argv, not a substring of this launcher's own command line."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            if any(a.endswith((b"/co_rl/train.py", b"/co_rl/sweep.py",
                                b"/co_rl/sweeps/stair_step_two_stage_mirror.sh")) for a in argv):
                found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return sorted(found)


def final_checkpoint(run, since, iterations):
    valid = []
    for marker in FINAL_ROOT.glob(f"*_{run}_s2/batch_complete.json"):
        if marker.stat().st_mtime < since:
            continue
        data = json.loads(marker.read_text())
        ckpt = marker.parent / data["checkpoint"]
        if (data["iterations_requested"] == iterations
                and data["last_iteration"] >= iterations - 1 and ckpt.is_file()):
            valid.append(ckpt)
    if not valid:
        raise RuntimeError("Local training stopped without a complete stage-2 checkpoint; migration not started")
    return max(valid, key=lambda p: p.stat().st_mtime)


def copy_project():
    # Isolated, new directory. Never overwrite another training project.
    remote(f"mkdir -p {shlex.quote(REMOTE_BASE)}; "
           f"mkdir {shlex.quote(REMOTE_REPO)}")
    args = ["tar", "-C", str(ROOT), "--exclude=./.git", "--exclude=__pycache__",
            "--exclude=*.egg-info", "--exclude=./.venv", "--exclude=./.private",
            "--exclude=./.env", "--exclude=./.env.*", "--exclude=./logs/co_rl/_runlogs/h200_queue",
            "-cf", "-", "."]
    sender = subprocess.Popen(args, stdout=subprocess.PIPE)
    try:
        receiver = subprocess.run(SSH + [f"tar -C {shlex.quote(REMOTE_REPO)} -xf -"], stdin=sender.stdout)
        sender.stdout.close()
        sender_rc = sender.wait()
        if sender_rc or receiver.returncode:
            raise RuntimeError(f"Project copy failed: tar={sender_rc}, ssh={receiver.returncode}")
    finally:
        if sender.poll() is None:
            sender.terminate()
            sender.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--wait-pid", type=int)
    source.add_argument("--completed-checkpoint", type=Path,
                        help="Resume a failed migration from an already completed local stage-2 run")
    p.add_argument("--run", default="two_stage")
    p.add_argument("--stage2-iterations", type=int, default=5000)
    p.add_argument("--poll-seconds", type=int, default=30)
    args = p.parse_args()
    state_dir = ROOT / "logs/co_rl/_runlogs/h200_queue"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = state_dir / "status.json"
    started = time.time()
    identity = proc_identity(args.wait_pid) if args.wait_pid is not None else None
    if args.wait_pid is not None and identity is None:
        raise RuntimeError("Expected local training supervisor is already absent; inspect completion manually")
    try:
        if args.completed_checkpoint is not None:
            if training_processes():
                raise RuntimeError("Local training is still active; cannot copy a live project")
            checkpoint = args.completed_checkpoint.resolve()
            expected = final_checkpoint(args.run, 0, args.stage2_iterations).resolve()
            if checkpoint != expected:
                raise RuntimeError("Resume checkpoint does not match the latest completed stage-2 run")
        else:
            write_status(status, "waiting_local_training", supervisor_pid=args.wait_pid,
                         remote_root=REMOTE_BASE, remote_writes_started=False)
            last_heartbeat = 0
            while proc_identity(args.wait_pid) == identity or training_processes():
                if time.monotonic() - last_heartbeat > 300:
                    # Keep ControlPersist alive while the local job takes several hours.
                    try:
                        remote("true", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except subprocess.CalledProcessError:
                        print("SSH heartbeat unavailable; local training is unaffected", flush=True)
                    last_heartbeat = time.monotonic()
                time.sleep(args.poll_seconds)
            checkpoint = final_checkpoint(args.run, started, args.stage2_iterations)
        log = (ROOT / f"logs/co_rl/_runlogs/{args.run}.log").read_text(errors="replace")
        if "stage 2 done: exit 0" not in log:
            raise RuntimeError("No successful stage-2 supervisor completion message")
        write_status(status, "copying", checkpoint=str(checkpoint), remote_root=REMOTE_BASE)
        if not (ROOT / STUDENT_CHECKPOINT).is_file():
            raise FileNotFoundError(f"Student checkpoint missing: {STUDENT_CHECKPOINT}")
        copy_project()
        relative_ckpt = checkpoint.relative_to(ROOT).as_posix()
        hashes = {}
        for relative in [relative_ckpt, STUDENT_CHECKPOINT.as_posix()]:
            with (ROOT / relative).open("rb") as stream:
                local_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            result = remote(f"sha256sum {shlex.quote(REMOTE_REPO + '/' + relative)}",
                            capture_output=True, text=True)
            if result.stdout.split()[0] != local_hash:
                raise RuntimeError(f"Transferred checkpoint checksum mismatch: {relative}")
            hashes[relative] = local_hash
        command = (f"cd {shlex.quote(REMOTE_REPO)} && "
                   "nohup bash scripts/co_rl/h200/bootstrap.sh " + shlex.quote(relative_ckpt) +
                   f" >{shlex.quote(REMOTE_BASE + '/bootstrap.log')} 2>&1 < /dev/null &")
        remote(command, stdout=subprocess.DEVNULL)
        write_status(status, "remote_bootstrap_started", remote_root=REMOTE_BASE,
                     checkpoint=relative_ckpt, checkpoint_sha256=hashes[relative_ckpt],
                     checkpoint_hashes=hashes,
                     note="Training begins only if simulator smoke test and scale benchmarks pass")
    except Exception as exc:
        write_status(status, "failed", error=str(exc), remote_root=REMOTE_BASE)
        raise


if __name__ == "__main__":
    main()
