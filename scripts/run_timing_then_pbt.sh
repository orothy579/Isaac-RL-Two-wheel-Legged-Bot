#!/usr/bin/env bash
# Chain: (1) wait for any running co_rl job, (2) finish the interrupted timing sweep
# (4 of 12 trials remain), (3) then launch PBT+adaptive automatically.
# Launch detached so it survives terminal/VSCode close (NOT a reboot):
#   nohup bash scripts/run_timing_then_pbt.sh > chain.out 2>&1 &
# NOTE: no `set -u` — Isaac Sim's conda setup script references unset vars (ZSH_VERSION)
cd /home/lch/Isaac-RL-Two-wheel-Legged-Bot

# conda env (activate.d scripts set the env vars isaacsim needs — a bare binary path fails)
source /home/lch/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -c "import isaacsim" || { echo "[chain] FATAL: isaacsim not importable — wrong env"; exit 1; }
echo "[chain] env OK ($(which python))"

# 1) wait for any already-running training/sweep/pbt
while pgrep -f "scripts/co_rl/(sweep|train|pbt)\.py" > /dev/null; do
    echo "[chain] $(date '+%F %T') another co_rl job is running — waiting..."
    sleep 300
done

# 2) resume the timing sweep to its full 12-trial budget (no-op if already complete)
TIMING_DIR=logs/co_rl/Flamingo_Light_Rough_Stair_Jump/ppo/_sweeps/stair_jump_timing_2026-07-15_10-43-58
echo "[chain] $(date '+%F %T') resuming timing sweep -> $TIMING_DIR"
python scripts/co_rl/sweep.py \
    --config scripts/co_rl/sweeps/stair_jump_timing.yaml \
    --results_dir "$TIMING_DIR" >> sweep_timing.out 2>&1
echo "[chain] $(date '+%F %T') timing sweep finished (exit $?)"

# 3) PBT + adaptive
echo "[chain] $(date '+%F %T') launching PBT"
python scripts/co_rl/pbt.py \
    --config scripts/co_rl/sweeps/pbt_stair_jump.yaml > pbt.out 2>&1
echo "[chain] $(date '+%F %T') PBT finished (exit $?)"
