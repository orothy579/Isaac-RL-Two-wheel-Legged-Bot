#!/usr/bin/env bash
set -e
source /home/lch/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
source /home/lch/isaacsim/setup_conda_env.sh
cd /home/lch/Isaac-RL-Two-wheel-Legged-Bot
# A separate user scope prevents an OOM in this replay from taking down the
# parent VS Code scope. Hard RAM/swap caps bound this job, not other user apps.
# CPU-only validation does not need a scope or GPU startup.
for argument in "$@"; do
    if [ "$argument" = "--validate-only" ]; then
        exec python -u scripts/co_rl/stair_oracle/replay_waypoints.py "$@"
    fi
done
exec systemd-run --user --scope --unit=stair-waypoint-replay \
    --property=MemoryHigh=5G --property=MemoryMax=6G \
    --property=MemorySwapMax=256M --property=OOMPolicy=stop \
    timeout --signal=TERM --kill-after=10s 600s \
    python -u scripts/co_rl/stair_oracle/replay_waypoints.py "$@"
