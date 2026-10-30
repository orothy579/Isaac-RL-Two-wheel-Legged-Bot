#!/usr/bin/env bash
# Called by after_current.py only after the complete local two-stage run.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
BASE=/home/work/postech/flamingo_h200
CKPT=${1:?relative checkpoint path required}
[[ "$REPO" == "$BASE/Isaac-RL-Two-wheel-Legged-Bot" ]]
mkdir -p "$BASE/cache" "$BASE/python" "$BASE/tmp"
export UV_CACHE_DIR="$BASE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$BASE/python"
export PIP_CACHE_DIR="$BASE/cache/pip"
export TMPDIR="$BASE/tmp"
export XDG_CACHE_HOME="$BASE/cache"
export HF_HOME="$BASE/cache/huggingface"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
cd "$REPO"
trap 'rc=$?; if (( rc != 0 )); then printf "{\"phase\":\"bootstrap_failed\",\"exit_code\":%d}\n" "$rc" > "$BASE/bootstrap_status.json"; fi' EXIT
printf '{"phase":"installing"}\n' > "$BASE/bootstrap_status.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$BASE/gpu.csv"
# ARM venvs cannot be copied to this x86_64 server. Rebuild in persistent storage.
uv python install 3.11
uv venv --python 3.11 "$BASE/venv"
PY="$BASE/venv/bin/python"
uv pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" 'isaaclab[isaacsim]==2.3.2.post1' --extra-index-url https://pypi.nvidia.com
uv pip install --python "$PY" 'numpy<2' tensordict==0.8.3 rsl-rl-lib==3.1.2 \
    tensorboard hydra-core gitpython imageio imageio-ffmpeg matplotlib scipy psutil reportlab pandas
uv pip install --python "$PY" --no-deps -e .
"$PY" -m pip freeze > "$BASE/installed_packages.txt" 2>/dev/null || uv pip freeze --python "$PY" > "$BASE/installed_packages.txt"
printf '{"phase":"validating_and_benchmarking"}\n' > "$BASE/bootstrap_status.json"
"$PY" -u scripts/co_rl/h200/dispatch.py --gain-checkpoint "$CKPT"
printf '{"phase":"training_complete"}\n' > "$BASE/bootstrap_status.json"
