#!/usr/bin/env bash
# Run a script of this repo with the Isaac Lab Python on the DGX Spark (GB10, aarch64;
# Isaac Sim 5.1 / Isaac Lab 2.3.2 pip install in /home/midbar/ws/isaaclab/env_isaaclab).
#   ./run_gb10.sh scripts/co_rl/train.py --task <task> --headless
#   ./run_gb10.sh scripts/co_rl/sweep.py --config scripts/co_rl/sweeps/<cfg>.yaml --python ./run_gb10.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${ISAACLAB_PY:-/home/midbar/ws/isaaclab/env_isaaclab/bin/python}"
# The pip isaaclab wheel ships isaaclab_tasks / isaaclab_rl / isaaclab_assets as sources only.
LAB_SRC="${ISAACLAB_SRC:-/home/midbar/ws/isaaclab/env_isaaclab/lib/python3.11/site-packages/isaaclab/source}"
# pandas (+tzdata) for sweep.py, installed --no-deps outside the shared venv.
PYDEPS="${FLAMINGO_PYDEPS:-/home/midbar/ws/isaaclab/flamingo_pydeps}"

# Only this repo on the path: the shell profile adds ROS Jazzy (Python 3.12) packages.
export PYTHONPATH="${ROOT}:${ROOT}/tools/gb10:${LAB_SRC}/isaaclab_tasks:${LAB_SRC}/isaaclab_rl:${LAB_SRC}/isaaclab_assets:${PYDEPS}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
# aarch64 (DGX Spark): Kit needs libgomp loaded before torch.
if [[ "$(uname -m)" == "aarch64" && -e /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
  case ":${LD_PRELOAD:-}:" in
    *":/lib/aarch64-linux-gnu/libgomp.so.1:"*) ;;
    *) export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/lib/aarch64-linux-gnu/libgomp.so.1" ;;
  esac
fi
cd "${ROOT}"
exec "${PY}" "$@"
