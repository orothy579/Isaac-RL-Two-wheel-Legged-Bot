#!/usr/bin/env bash
# Keep the long training lanes alive until Nov 2 (Claude, 2026-10-31): every 2 minutes, a lane whose trainer is gone
# before reaching its total iteration count is restarted from its own latest.pt for the remaining iterations.
# Memory guard (user: at most 70 % load): above 70 % of the container's memory limit the lowest-priority running lane
# is stopped (SIGTERM: it saves latest.pt) and stays paused until memory is back under 60 %.
#   lane = <name>:<script>:<out dir>:<total iterations>:<extra args, ',' for ' '>, highest priority first
set -uo pipefail
BASE=/home/work/postech/flamingo_light
REPO=$BASE/Isaac-RL-Two-wheel-Legged-Bot
LOG=$BASE/claude_queue/lanes
SITE=$("$BASE/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cd "$REPO"
log() { echo "[$(date -Iseconds)] $*" >> "$LOG/supervisor.log"; }

done_iters() {  # <out dir> -> iterations finished (latest.pt iteration + 1), 0 without one
    CUDA_VISIBLE_DEVICES= "$BASE/venv/bin/python" -c "
import os, sys, torch
p = os.path.join(sys.argv[1], 'latest.pt')
print(torch.load(p, map_location='cpu', weights_only=False)['iteration'] + 1 if os.path.isfile(p) else 0)" "$1" 2>/dev/null
}

MEM_MAX=$(cat /sys/fs/cgroup/memory.max)
mem_pct() { echo $(( $(cat /sys/fs/cgroup/memory.current) * 100 / MEM_MAX )); }
trainer_pid() { pgrep -f "[${1:0:1}]${1:1} --init .* --out $2( |$)" | head -1; }
declare -A PAUSED

log "start: $*"
while true; do
    if (( $(mem_pct) > 70 )); then  # stop the lowest-priority running lane
        for ((i = $#; i >= 1; i--)); do
            IFS=: read -r name script out total extra <<< "${!i}"
            p=$(trainer_pid "$script" "$out")
            if [[ -n $p ]]; then
                log "$name: memory $(mem_pct) % > 70 % -> paused (SIGTERM $p)"
                kill -TERM "$p"
                PAUSED[$name]=1
                break
            fi
        done
    fi
    for lane in "$@"; do
        IFS=: read -r name script out total extra <<< "$lane"
        if [[ -n $(trainer_pid "$script" "$out") ]]; then
            continue
        fi
        if [[ -n ${PAUSED[$name]:-} ]]; then
            (( $(mem_pct) < 60 )) || continue
            unset "PAUSED[$name]"
            log "$name: memory $(mem_pct) % < 60 % -> resumed"
        fi
        n=$(done_iters "$out")
        left=$((total - n))
        if ((left <= 0)); then
            continue
        fi
        log "$name: trainer not running, $n of $total done -> restart from $out/latest.pt for $left"
        (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES XDG_CACHE_HOME=$BASE/cache \
            TMPDIR=$BASE/tmp PYTHONPATH=.:$SITE/isaaclab/source/isaaclab_tasks:$SITE/isaaclab/source/isaaclab_rl:$SITE/isaaclab/source/isaaclab_assets \
            setsid nohup "$BASE/venv/bin/python" -u "scripts/co_rl/$script" --init "$out/latest.pt" --out "$out" \
            --iterations "$left" ${extra//,/ } >> "$LOG/${name}_restarts.log" 2>&1 < /dev/null &)
        sleep 60
    done
    sleep 120
done
