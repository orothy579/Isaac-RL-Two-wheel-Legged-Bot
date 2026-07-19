#!/usr/bin/env bash
# Chain: wait for the running PBT round-2 (and any other co_rl job), then launch the
# fixed-height jump capability probe (10..15 cm, binary can/can't verdicts).
# Launch detached so it survives terminal/VSCode close (NOT a reboot):
#   nohup bash scripts/run_pbt_then_probe.sh > probe_chain.out 2>&1 &
# NOTE: no `set -u` — Isaac Sim's conda setup script references unset vars (ZSH_VERSION)
cd /home/lch/Isaac-RL-Two-wheel-Legged-Bot

# conda env (activate.d scripts set the env vars isaacsim needs — a bare binary path fails)
source /home/lch/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python -c "import isaacsim" || { echo "[chain] FATAL: isaacsim not importable — wrong env"; exit 1; }
echo "[chain] env OK ($(which python))"

# 1) wait for any already-running training/sweep/pbt (probe_jump is NOT in the pattern)
while pgrep -f "scripts/co_rl/(sweep|train|pbt)\.py" > /dev/null; do
    echo "[chain] $(date '+%F %T') another co_rl job is running — waiting..."
    sleep 300
done

# 2) capability probe: 10..15 cm fixed-height steps, generous budget, early-stop on PASS
echo "[chain] $(date '+%F %T') launching jump capability probe"
python scripts/co_rl/probe_jump.py --iters 8000 > probe.out 2>&1
echo "[chain] $(date '+%F %T') probe finished (exit $?)"
