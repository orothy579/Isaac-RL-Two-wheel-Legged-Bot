"""Run one simulator with the existing shared lock; invoke in a capped systemd service."""
import argparse
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.co_rl.stair_oracle.replay_guard import acquire_replay_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('a command is required')
    lock = acquire_replay_lock(ROOT / 'scripts/co_rl/stair_oracle/waypoint_runs/.replay.lock')
    child = subprocess.Popen(args.command, cwd=ROOT, start_new_session=True)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        code = child.wait(timeout=args.seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os_signal = signal.SIGTERM
        import os
        os.killpg(child.pid, os_signal)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        code = 124
    finally:
        lock.close()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
