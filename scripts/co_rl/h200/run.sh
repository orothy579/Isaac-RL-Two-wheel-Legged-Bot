#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
BASE=/home/work/postech/flamingo_light
PY="$BASE/venv/bin/python"
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
SRC="$SITE/isaaclab/source"
export PYTHONPATH="$ROOT:$SRC/isaaclab_tasks:$SRC/isaaclab_rl:$SRC/isaaclab_assets"
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="$BASE/cache"
export HF_HOME="$BASE/cache/huggingface"
export TMPDIR="$BASE/tmp"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "$ROOT"
exec "$PY" "$@"
