#!/usr/bin/env bash
# One memory-capped job. All commands run in the existing Isaac Sim environment.
set -euo pipefail
label=$1
shift
[[ "$label" =~ ^[a-z0-9-]+$ ]] || exit 2
wall_seconds=${FLAMINGO_JOB_SECONDS:-3600}
[[ "$wall_seconds" =~ ^[0-9]+$ ]] && (( wall_seconds > 0 && wall_seconds <= 28800 )) || exit 2
root=/home/lch/Isaac-RL-Two-wheel-Legged-Bot
mkdir -p "$root/outputs/retrain_20260909"
exec systemd-run --user --unit="flamingo-$label" \
    --property=MemoryHigh=6G --property=MemoryMax=7G \
    --property=MemorySwapMax=256M --property=OOMPolicy=stop \
    --property="RuntimeMaxSec=$((wall_seconds + 100))" --property=TimeoutStopSec=15 \
    --property="StandardOutput=append:$root/outputs/retrain_20260909/$label.log" \
    --property=StandardError=inherit \
    /bin/bash "$root/scripts/co_rl/staged_retrain/runtime.sh" \
    python -u scripts/co_rl/staged_retrain/run_bounded.py --seconds "$wall_seconds" "$@"
