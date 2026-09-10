#!/usr/bin/env bash
set -eo pipefail
source /home/lch/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
source /home/lch/isaacsim/setup_conda_env.sh
cd /home/lch/Isaac-RL-Two-wheel-Legged-Bot
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
