#!/usr/bin/env bash
# H200 queue toward the deployable goal (Claude, 2026-10-31): a BLIND policy that climbs >= 1 riser of the
# 15 cm row in >= 50 % of episodes, in Isaac and in MuJoCo. One job at a time on the single H200 (a second
# process time-slices the GPU and slows both). Starts when the running chain (claude_resume_chain.sh, its
# Codex student job) has released the GPU. Expected wall times (blind PPO 65536 envs ~12.8 s/it):
#
#   S1  eval student_ft model_2000/2500/2999 (blind, 256 env, 9.3-15 cm)                    ~0.2 h
#   S2  DAgger re-distillation from the new teacher (H200 model_2749, 15 cm 90 %), 4096 env  ~1.3 h
#   S3  PPO fine-tune of that student, 65536 env, 2000 it + evals                            ~7.4 h
#   S4  robust fine-tune (wider DR) of the best blind checkpoint of S1/S3, 2500 it + evals   ~9.4 h
#   S5  robust fine-tune of the best checkpoint of the other lineage, 2500 it + evals        ~9.4 h
#   S6  continue the better robust run, 3000 it + evals                                      ~11.1 h
#   S7  continue it again, 4000 it + evals (fills Nov 2)                                     ~14.7 h
#
# Iteration cap: no weights may have more than $CAP (15000) training iterations behind them, counted over the
# whole lineage - the DAgger student's 1500 plus every PPO update of each run warm-started or resumed from it.
# Stages are shortened, or skipped, to stay under it. With the lengths above the longest lineage ends at 14000
# (DAgger 1500 + student_ft 3000 + robust 2500 + 3000 + 4000); the teacher model_2749 has about 12250 behind it.
#
# Every checkpoint is evaluated with scripts/co_rl/eval_stair_step.py (nominal play task; robust runs also on
# the robust play task); scripts/co_rl/sweeps/pick_best_blind.py picks by the 15 cm share. A failed stage is
# logged and the queue goes on with what exists. Progress: $Q/status.json, $Q/queue.log, $Q/summary.txt.
set -uo pipefail
BASE=/home/work/postech/flamingo_light
REPO=$BASE/Isaac-RL-Two-wheel-Legged-Bot
Q=${Q:-$BASE/claude_queue}
mkdir -p "$Q"
cd "$REPO"

G=${G:-logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift_Mirror_Gain/ppo}
DIST=${DIST:-logs/co_rl/Flamingo_Light_Rough_Stair_Step_Distill}
BLIND=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-v1-ppo
BLIND_PLAY=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Play-v1-ppo
ROBUST=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Robust-v1-ppo
ROBUST_PLAY=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Robust-Play-v1-ppo
PERC=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-v1-ppo
TEACHER=${TEACHER:-$G/2026-10-30_13-12-36_h200_scaled_resume500_n131072/model_2749.pt}
STUDENT_FT=${STUDENT_FT:-$G/2026-10-31_05-27-47_student_ft_1793390685}
KIT="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
N=65536
WAIT_PID=${WAIT_PID:-31149}  # claude_resume_chain.sh
CAP=${CAP:-15000}
# run dir -> training iterations behind it, such that its model_N has BASE_OF + N + 1 behind it
declare -A BASE_OF
BASE_OF[$STUDENT_FT]=1500  # warm-started from the 1500-iteration DAgger student (teacher Gain 2999)

log() { echo "[$(date -Iseconds)] $*" | tee -a "$Q/queue.log"; }
status() {
    printf '{"stage": "%s", "detail": "%s", "updated": "%s", "pid": %d}\n' "$1" "$2" "$(date -Iseconds)" $$ > "$Q/status.tmp"
    mv "$Q/status.tmp" "$Q/status.json"
    log "$1: $2"
}

overrides() {  # num_envs -> JSON file; same capacity formula as scripts/co_rl/h200/scale_train.py
    local f=$Q/overrides_$1.json
    python3 - "$1" > "$f" <<'EOF'
import json, math, sys
n = int(sys.argv[1]); factor = math.ceil(n / 4096)
print(json.dumps({"env.sim.device": "cuda:0", "agent.device": "cuda:0",
                  "agent.algorithm.num_mini_batches": max(4, n // 2048),
                  "env.sim.physx.gpu_max_rigid_patch_count": max(5 * 2**15, n * 32),
                  "env.sim.physx.gpu_found_lost_pairs_capacity": max(2**21, n * 64),
                  "env.sim.physx.gpu_heap_capacity": 2**26 * min(factor, 8),
                  "env.sim.physx.gpu_collision_stack_size": 2**26 * min(factor, 4)}, indent=2))
EOF
    echo "$f"
}

dry_run() {  # DRY_RUN=1: fake a job's outputs, to check the queue's control flow without the GPU
    python3 "$REPO/scripts/co_rl/sweeps/queue_dry_run.py" "$G" "$@"
}

gpu() {  # <job name> <timeout s> <script> <args...>: waits for Codex's GPU lock, one job at a time
    local name=$1 tmo=$2
    shift 2
    log "start $name"
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        dry_run "$@"
        log "end $name rc=$? (dry run)"
        return 0
    fi
    flock "$BASE/gpu_0.lock" timeout "$tmo" bash scripts/co_rl/h200/run.sh -u "$@" > "$Q/$name.log" 2>&1
    local rc=$?
    log "end $name rc=$rc"
    return $rc
}

train() {  # <run name> <task> <iterations> warm <ckpt> | resume <run dir name> <ckpt file>
    local name=$1 task=$2 iters=$3 mode=$4
    shift 4
    local src
    if [[ $mode == warm ]]; then
        src=(--warmstart_ckpt "$1" --warmstart_std_scale 1.0)
    else
        src=(--resume True --load_run "$1" --checkpoint "$2")
    fi
    gpu "train_$name" "$((iters * 20 + 1800))" scripts/co_rl/train.py --task "$task" --algo ppo --headless \
        --num_envs "$N" --device cuda:0 --num_policy_stacks 49 --num_critic_stacks 2 --max_iterations "$iters" \
        --batch_exit --run_name "$name" --param_overrides "$(overrides "$N")" "${src[@]}" --kit_args "$KIT"
}

run_dir() { ls -d "$G"/*_"$1" 2>/dev/null | tail -1; }

EVALS_OUT=""
evaluate() {  # <ckpt> <tag> <play task>: writes <run dir>/eval_<tag>_it<N>_256.json, sets EVALS_OUT
    local ck=$1 tag=$2 task=$3
    local it
    it=$(basename "$ck" .pt | sed 's/^model_//; s/^student_//')
    local out
    out=$(dirname "$ck")/eval_${tag}_it${it}_256.json
    EVALS_OUT=""
    [[ -f $ck ]] || { log "skip eval: no $ck"; return 1; }
    if [[ ! -f $out ]]; then
        gpu "eval_$(basename "$(dirname "$ck")")_${tag}_$it" 1800 scripts/co_rl/eval_stair_step.py --task "$task" \
            --num_policy_stacks 49 --checkpoint "$ck" --levels 0 3 6 9 --num_envs 256 --seconds 15 --out "$out" || return 1
    fi
    EVALS_OUT=$out
}

eval_run() {  # <run dir> <robust: 0|1> <iteration>...: appends the JSON paths to the array named by $EV
    local dir=$1 robust=$2
    shift 2
    local it
    for it in "$@"; do
        evaluate "$dir/model_$it.pt" nominal "$BLIND_PLAY" && eval "$EV+=(\"$EVALS_OUT\")"
        if [[ $robust == 1 ]]; then
            evaluate "$dir/model_$it.pt" robust "$ROBUST_PLAY" && eval "$EV+=(\"$EVALS_OUT\")"
        fi
    done
}

best_of() { python3 scripts/co_rl/sweeps/pick_best_blind.py "$@" 2>/dev/null; }
last_ckpt() { ls "$1"/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
ckpt_iter() { local n; n=$(basename "$1" .pt); echo "${n##*_}"; }

cum_of() {  # <ckpt> -> training iterations behind these weights (empty if the lineage is unknown)
    local dir n
    dir=$(dirname "$1")
    n=$(ckpt_iter "$1")
    if [[ $(basename "$1") == student_* ]]; then
        echo $((n + 1))  # DAgger student_N: N + 1 iterations
    elif [[ -n ${BASE_OF[$dir]+x} ]]; then
        echo $((BASE_OF[$dir] + n + 1))
    fi
}

capped() {  # <iterations behind the start weights> <planned> -> what $CAP allows (<= 0: nothing)
    local left=$((CAP - $1))
    echo $((left < $2 ? left : $2))
}

eval_points() {  # <last iteration> -> the last two multiples of 500 (1000 from 3000 on) below it, and it
    local last=$1 step=500 k pts=()
    ((last >= 3000)) && step=1000
    for ((k = step; k < last; k += step)); do pts+=("$k"); done
    local s=$((${#pts[@]} > 2 ? ${#pts[@]} - 2 : 0))
    echo "${pts[@]:s}" "$last"
}

robust_stage() {  # <stage> <run name> <start ckpt>: warm-started robust fine-tune, capped, then evaluated
    local stage=$1 name=$2 start=$3 used k dir
    used=$(cum_of "$start")
    [[ -z $used ]] && { log "$stage: lineage of $start unknown, skipped"; return; }
    k=$(capped "$used" 2500)
    ((k > 0)) || { log "$stage: no iterations left under the cap ($used of $CAP behind $start)"; return; }
    status "$stage" "robust fine-tune of $start ($used it behind it) for $k it"
    train "$name" "$ROBUST" "$k" warm "$start"
    dir=$(run_dir "$name")
    [[ -z $dir ]] && return
    BASE_OF[$dir]=$used
    EV=EV_R eval_run "$dir" 1 $(eval_points $((k - 1)))
}

# ------------------------------------------------------------------------------------------------------------
status wait "for pid $WAIT_PID (the running chain) to finish"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done

# S1: the running blind PPO fine-tune (student_ft, from the Gain-2999 DAgger student)
status S1 "eval student_ft"
EV_A=()
[[ -f $STUDENT_FT/eval_it1400_256.json ]] && EV_A+=("$STUDENT_FT/eval_it1400_256.json")
EV=EV_A eval_run "$STUDENT_FT" 0 2000 2500 2999

# S2: DAgger re-distillation from the H200 teacher
status S2 "distill from $TEACHER"
DDIR=$DIST/$(date +%Y-%m-%d_%H-%M-%S)_teacher2749_h50
gpu distill_2749 14400 scripts/co_rl/distill_stair_step.py --task "$PERC" --teacher_ckpt "$TEACHER" \
    --num_envs 4096 --iterations 1500 --out_dir "$DDIR"
S2CK=$DDIR/student_1499.pt
EV_B=()
evaluate "$S2CK" nominal "$BLIND_PLAY" && EV_B+=("$EVALS_OUT")

# S3: PPO fine-tune of the new student
if [[ -f $S2CK ]]; then
    used=$(cum_of "$S2CK")
    k=$(capped "$used" 2000)
    if ((k > 0)); then
        status S3 "PPO fine-tune of $S2CK ($used it behind it) for $k it"
        train student2749_ft "$BLIND" "$k" warm "$S2CK"
        RB=$(run_dir student2749_ft)
        if [[ -n $RB ]]; then
            BASE_OF[$RB]=$used
            EV=EV_B eval_run "$RB" 0 $(eval_points $((k - 1)))
        fi
    fi
fi

# S4 / S5: robust fine-tunes of the best checkpoint of each lineage, the overall best first
BEST_A=$(best_of "${EV_A[@]}")
BEST_B=$( ((${#EV_B[@]})) && best_of "${EV_B[@]}")
FIRST=$(best_of "${EV_A[@]}" "${EV_B[@]}")
if [[ $FIRST == "$BEST_A" ]]; then SECOND=$BEST_B; else SECOND=$BEST_A; fi
EV_R=()
[[ -n $FIRST ]] && robust_stage S4 robustA "$FIRST"
[[ -n $SECOND ]] && robust_stage S5 robustB "$SECOND"

# S6 / S7: continue the better robust run (nominal and robust play averaged), capped
for stage in S6:3000 S7:4000; do
    WIN=$( ((${#EV_R[@]})) && best_of "${EV_R[@]}")
    [[ -z $WIN ]] && break
    used=$(cum_of "$WIN")
    [[ -z $used ]] && { log "${stage%%:*}: lineage of $WIN unknown, stopped"; break; }
    # a resume from model_M runs M .. M + k - 1: k more iterations on top of the "used" ones
    k=$(capped "$used" "${stage##*:}")
    ((k > 0)) || { log "${stage%%:*}: no iterations left under the cap ($used of $CAP behind $WIN)"; break; }
    name=robust_cont_${stage%%:*}
    status "${stage%%:*}" "continue $WIN ($used it behind it) for $k it"
    train "$name" "$ROBUST" "$k" resume "$(basename "$(dirname "$WIN")")" "$(basename "$WIN")"
    RC=$(run_dir "$name")
    [[ -z $RC ]] && break
    M=$(ckpt_iter "$WIN")
    BASE_OF[$RC]=$((used - M))  # its model_j (j >= M) has used + (j - M + 1) behind it
    EV=EV_R eval_run "$RC" 1 $(eval_points "$(last_ckpt "$RC")")
done

python3 scripts/co_rl/sweeps/pick_best_blind.py --table "${EV_A[@]}" "${EV_B[@]}" "${EV_R[@]}" > "$Q/summary.txt" 2>&1
BEST=$(best_of "${EV_A[@]}" "${EV_B[@]}" "${EV_R[@]}")
{
    echo
    echo "best (15 cm, nominal + robust averaged where both exist): $BEST ($(cum_of "$BEST") training iterations behind it, cap $CAP)"
    echo "training iterations behind model_0 of each run (model_N: + N + 1):"
    for d in "${!BASE_OF[@]}"; do echo "  $d  ${BASE_OF[$d]}"; done
} >> "$Q/summary.txt"
status done "see $Q/summary.txt"
