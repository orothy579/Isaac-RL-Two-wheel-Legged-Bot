#!/usr/bin/env bash
# Evaluate the parallel training lanes as they go (Claude, 2026-10-31): every new checkpoint at least STEP iterations
# past the last evaluated one of its lane is scored in Isaac (eval_stair_step.py, nominal blind play task, 256 envs)
# and in the MuJoCo cosim (cosim_eval_policy.py, Isaac's speed limits, 1024 trials). The goal is 15 cm, 3 consecutive
# risers >= 70 % in BOTH simulators; $LOG/ranking.txt lists every scored checkpoint by the lower of the two.
#   lanes: "<run dir>:<kind>" with kind = co_rl (model_N.pt, actor exported here) or npz (model_N.pt + actor_N.npz)
set -uo pipefail
BASE=/home/work/postech/flamingo_light
REPO=$BASE/Isaac-RL-Two-wheel-Legged-Bot
LOG=${LOG:-$BASE/claude_queue/lanes}
STEP=${STEP:-100}
MJ_WORKERS=${MJ_WORKERS:-4}
PLAY=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Play-v1-ppo
cd "$REPO"
declare -A LAST

log() { echo "[$(date -Iseconds)] $*" >> "$LOG/eval_watch.log"; }

eval_ckpt() {  # <run dir> <kind> <iteration>
    local dir=$1 kind=$2 it=$3 ck npz
    ck=$dir/model_$it.pt
    if [[ $kind == co_rl ]]; then
        npz=$dir/actor_it$it.npz
        [[ -f $npz ]] || CUDA_VISIBLE_DEVICES= "$BASE/venv/bin/python" cosim-anaylze/tools/export_co_rl_actor.py "$ck" "$npz" >/dev/null 2>&1
    else
        npz=$dir/actor_$it.npz
    fi
    if [[ ! -f $dir/eval_nominal_it${it}_256.json ]]; then
        log "isaac $ck"
        timeout 2400 bash scripts/co_rl/h200/run.sh -u scripts/co_rl/eval_stair_step.py --task "$PLAY" \
            --num_policy_stacks 49 --checkpoint "$ck" --levels 0 3 6 9 --num_envs 256 --seconds 15 \
            --out "$dir/eval_nominal_it${it}_256.json" > "$LOG/eval_isaac_$(basename "$dir")_$it.log" 2>&1
    fi
    if [[ ! -f $dir/cosim_vlim_it${it}_256.json && -f $npz ]]; then
        log "mujoco $npz"
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 3600 "$BASE/cosim_venv/bin/python" scripts/co_rl/cosim_eval_policy.py \
            --actor "$npz" --checkpoint "$ck" --frames 50 --no_scan --levels 0 3 6 9 --trials 256 \
            --workers "$MJ_WORKERS" --out "$dir/cosim_vlim_it${it}_256.json" > "$LOG/eval_mujoco_$(basename "$dir")_$it.log" 2>&1
    fi
}

ranking() {
    python3 - "$LOG/ranking.txt" "$@" <<'EOF'
import glob, json, os, sys
rows = []
for d in sys.argv[2:]:
    for f in glob.glob(os.path.join(d, "eval_nominal_it*_256.json")):
        it = f.split("_it")[-1].split("_")[0]
        g = os.path.join(d, f"cosim_vlim_it{it}_256.json")
        try:
            i9 = json.load(open(f))["levels"]["9"]
            m9 = json.load(open(g))["levels"]["9"] if os.path.isfile(g) else None
        except Exception:
            continue
        mi, mm = i9.get("envs_climbed_3plus", 0.0), (m9 or {}).get("envs_climbed_3plus", 0.0)
        rows.append((min(mi, mm), mi, i9.get("risers_climbed_mean", 0.0), mm,
                     (m9 or {}).get("risers_climbed_mean", 0.0), m9 is not None, os.path.join(d, f"model_{it}.pt")))
rows.sort(reverse=True)
with open(sys.argv[1], "w") as out:
    out.write("15 cm, 3 consecutive risers / mean risers (min of the two first)\n")
    for r in rows:
        out.write(f"min {r[0]*100:5.1f} %  isaac {r[1]*100:5.1f} % / {r[2]:.2f}  mujoco "
                  f"{(f'{r[3]*100:5.1f} % / {r[4]:.2f}' if r[5] else 'pending')}  {r[6]}\n")
scored = [r for r in rows if r[5]]
if scored:  # the best checkpoint scored in both simulators
    best = max(scored, key=lambda r: (r[0], r[1] + r[3], r[2] + r[4]))
    json.dump({"checkpoint": best[6], "min_3plus_15cm": best[0], "isaac_3plus": best[1], "isaac_mean": best[2],
               "mujoco_3plus": best[3], "mujoco_mean": best[4]}, open(os.path.join(os.path.dirname(sys.argv[1]), "best.json"), "w"), indent=1)
EOF
}

export_best() {  # keep $LOG/best_blind.onnx = the current best checkpoint
    local ck
    ck=$(python3 -c "import json; print(json.load(open('$LOG/best.json'))['checkpoint'])" 2>/dev/null) || return
    [[ -n $ck && $ck != "${EXPORTED:-}" ]] || return
    CUDA_VISIBLE_DEVICES= "$BASE/venv/bin/python" scripts/co_rl/export_blind_onnx.py "$ck" "$LOG/best_blind.onnx" \
        > "$LOG/best_blind_onnx.log" 2>&1 && cp "$LOG/best.json" "$LOG/best_blind_onnx.json" && EXPORTED=$ck && log "onnx <- $ck"
}

log "start: lanes $*"
while true; do
    for lane in "$@"; do
        dir=${lane%%:*} kind=${lane##*:}
        [[ -d $dir ]] || continue
        newest=$(ls "$dir"/model_*.pt 2>/dev/null | sed -n 's/.*model_\([0-9]*\)\.pt/\1/p' | sort -n | tail -1)
        [[ -z $newest ]] && continue
        last=${LAST[$dir]:--1}
        if (( last < 0 || newest - last >= STEP )); then
            eval_ckpt "$dir" "$kind" "$newest"
            LAST[$dir]=$newest
            ranking "${@%%:*}"
            export_best
        fi
    done
    sleep 120
done
