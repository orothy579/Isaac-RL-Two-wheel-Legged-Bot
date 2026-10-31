#!/usr/bin/env bash
# H200 queue toward the deployable goal (Claude, 2026-10-31): a BLIND policy that climbs 3 consecutive risers of
# the 15 cm row in >= 50 % of episodes, in Isaac and in MuJoCo. One job at a time on the single H200 (a second
# process time-slices the GPU and slows both). Starts when the running chain (claude_resume_chain.sh, its
# Codex student job) has released the GPU. Expected wall times (blind PPO 65536 envs ~12.8 s/it):
#
#   S1  eval student_ft model_1400 .. 2999 (10 checkpoints, blind, 256 env, 9.3-15 cm)        ~0.7 h
#   S2  DAgger re-distillation from the new teacher (H200 model_2749, 15 cm 90 %), 4096 env  ~1.3 h
#   S3  PPO fine-tune of that student, 65536 env, 2000 it + evals                            ~7.4 h
#   S4  A: Robust-Reset task from the best blind checkpoint of S1-S3 (diverse resets from a  ~9.4 h
#       state bank, lift-after-riser-contact reward, jumping allowed), 2500 it + evals
#   S5  B: the plain Robust task from the SAME checkpoint, 2500 it + evals                  ~9.4 h
#   S6  continue the better of A / B (same task), 3000 it + evals                            ~11.1 h
#   S7  continue it again, 4000 it + evals (fills Nov 2)                                     ~14.7 h
#
# Speed limits: the robot's own (user, 2026-10-31: no retraining for a 0.5 m/s cap). Isaac clamps the shoulder,
# leg and wheel joints at the USD's maxJointVelocity (40 rad/s = 3.04 m/s rim speed) and the wheel targets are
# +-40 rad/s (action +-1 x scale 40); the MuJoCo evaluation applies the same joint clamps
# (scripts/co_rl/cosim_eval_policy.py, cosim_vlim_*.json). A 0.5 m/s clip of the wheel targets was tried and
# dropped: with the velocity-damping wheel drive (0.3 x (target - speed)) it also cut the torque to <= 2 N m, and
# every policy fell over within 0.6 s, before reaching the stairs.
#
# Iteration cap: no weights may have more than $CAP (15000) training iterations behind them, counted over the
# whole lineage - the DAgger student's 1500 plus every PPO update of each run warm-started or resumed from it.
# Stages are shortened, or skipped, to stay under it. With the lengths above the longest lineage ends at 14000
# (DAgger 1500 + student_ft 3000 + robust 2500 + 3000 + 4000); the teacher model_2749 has about 12250 behind it.
#
# Every checkpoint is evaluated in Isaac with scripts/co_rl/eval_stair_step.py (nominal play task; robust runs
# also on the robust play task) and in the MuJoCo cosim with scripts/co_rl/cosim_eval_policy.py (CPU, in the
# background, one at a time; its blind input matches Isaac's to 2e-7). scripts/co_rl/sweeps/pick_best_blind.py
# picks by the 15 cm 3-riser share averaged over the two simulators. A failed stage is logged and the queue goes
# on with what exists. Progress: $Q/status.json, $Q/queue.log, $Q/summary.txt.
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
ROBUST_RESET=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Robust-Reset-v1-ppo
BANK=${BANK:-logs/co_rl/_reset_bank/stair_bank_v1.npz}  # the Robust-Reset task's state bank
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
            --num_policy_stacks 49 --checkpoint "$ck" --levels 0 3 6 9 --num_envs 256 --seconds 15 \
            --out "$out" || return 1
    fi
    EVALS_OUT=$out
}

COSIM_PIDS=()
COSIM_OUT=""
cosim_eval() {  # <ckpt>: export its actor and score it in the MuJoCo cosim (CPU) in the background, one at a time
    local ck=$1 it out npz
    it=$(ckpt_iter "$ck")
    out=$(dirname "$ck")/cosim_vlim_it${it}_256.json  # with Isaac's joint speed limits
    npz=$(dirname "$ck")/actor_it${it}.npz
    COSIM_OUT=$out
    [[ -f $out ]] && return 0
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        python3 "$REPO/scripts/co_rl/sweeps/queue_dry_run.py" "$G" cosim_eval_policy.py --checkpoint "$ck" --out "$out"
        return 0
    fi
    (
        flock 9
        CUDA_VISIBLE_DEVICES= "$BASE/venv/bin/python" cosim-anaylze/tools/export_co_rl_actor.py "$ck" "$npz" &&
            "$BASE/cosim_venv/bin/python" scripts/co_rl/cosim_eval_policy.py --actor "$npz" --checkpoint "$ck" \
                --frames 50 --no_scan --levels 0 3 6 9 --trials 256 --workers 8 --out "$out"
    ) 9> "$Q/cosim.lock" > "$Q/cosim_$(basename "$(dirname "$ck")")_$it.log" 2>&1 &
    COSIM_PIDS+=("$!")
}
cosim_wait() {  # before a selection: every pending MuJoCo score must be in
    local p
    for p in "${COSIM_PIDS[@]}"; do wait "$p"; done
    COSIM_PIDS=()
}

eval_run() {  # <run dir> <robust: 0|1> <iteration>...: appends the JSON paths (Isaac, MuJoCo) to the array named by $EV
    local dir=$1 robust=$2
    shift 2
    local it
    for it in "$@"; do
        evaluate "$dir/model_$it.pt" nominal "$BLIND_PLAY" || continue
        eval "$EV+=(\"$EVALS_OUT\")"
        if [[ $robust == 1 ]]; then
            evaluate "$dir/model_$it.pt" robust "$ROBUST_PLAY" && eval "$EV+=(\"$EVALS_OUT\")"
        fi
        cosim_eval "$dir/model_$it.pt"
        eval "$EV+=(\"$COSIM_OUT\")"
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

eval_points() {  # <first> <last> <step> -> every multiple of step in (first, last), and last (step: a multiple
    # of the 100-iteration save interval)
    # dense on purpose: the deterministic 15 cm result jumps between neighbouring checkpoints while the training
    # curves stay flat (teacher model_400 61.7 % -> model_500 40.6 %, student 1400 43.4 % -> 1700 19.9 %)
    local first=$1 last=$2 step=$3 k pts=()
    for ((k = (first / step + 1) * step; k < last; k += step)); do pts+=("$k"); done
    echo "${pts[@]}" "$last"
}

declare -A TASK_OF  # robust run dir -> its training task (a continuation keeps it)
robust_stage() {  # <stage> <run name> <task> <start ckpt>: warm-started robust fine-tune, capped, then evaluated
    local stage=$1 name=$2 task=$3 start=$4 used k dir
    used=$(cum_of "$start")
    [[ -z $used ]] && { log "$stage: lineage of $start unknown, skipped"; return; }
    k=$(capped "$used" 2500)
    ((k > 0)) || { log "$stage: no iterations left under the cap ($used of $CAP behind $start)"; return; }
    status "$stage" "$task from $start ($used it behind it) for $k it"
    train "$name" "$task" "$k" warm "$start"
    dir=$(run_dir "$name")
    [[ -z $dir ]] && return
    BASE_OF[$dir]=$used
    TASK_OF[$dir]=$task
    EV=EV_R eval_run "$dir" 1 $(eval_points 0 $((k - 1)) 500)
}

# ------------------------------------------------------------------------------------------------------------
status wait "for pid $WAIT_PID (the running chain) to finish"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done

# S1: the running blind PPO fine-tune (student_ft, from the Gain-2999 DAgger student)
status S1 "eval student_ft"
EV_A=()
EV=EV_A eval_run "$STUDENT_FT" 0 1400 1500 1600 1700 1800 2000 2200 2500 2700 2999  # saved every 100 it

# S2: DAgger re-distillation from the H200 teacher
status S2 "distill from $TEACHER"
DDIR=$DIST/$(date +%Y-%m-%d_%H-%M-%S)_teacher2749_h50
gpu distill_2749 14400 scripts/co_rl/distill_stair_step.py --task "$PERC" --teacher_ckpt "$TEACHER" \
    --num_envs 4096 --iterations 1500 --out_dir "$DDIR"
S2CK=$DDIR/student_1499.pt
EV_B=()
if evaluate "$S2CK" nominal "$BLIND_PLAY"; then
    EV_B+=("$EVALS_OUT")
    cosim_eval "$S2CK"
    EV_B+=("$COSIM_OUT")
fi

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
            EV=EV_B eval_run "$RB" 0 $(eval_points 0 $((k - 1)) 300)
        fi
    fi
fi

# S4 / S5: A/B from the same start - the best blind checkpoint so far (both simulators)
cosim_wait
FIRST=$(best_of "${EV_A[@]}" "${EV_B[@]}")
EV_R=()
if [[ -n $FIRST ]]; then
    if [[ -f $BANK ]]; then
        robust_stage S4 robust_reset "$ROBUST_RESET" "$FIRST"
    else
        log "S4: no state bank $BANK, the Robust-Reset arm is skipped"
    fi
    robust_stage S5 robust_plain "$ROBUST" "$FIRST"
fi

# S6 / S7: continue the better robust run (nominal and robust play averaged), capped
for stage in S6:3000 S7:4000; do
    cosim_wait
    WIN=$( ((${#EV_R[@]})) && best_of "${EV_R[@]}")
    [[ -z $WIN ]] && break
    used=$(cum_of "$WIN")
    [[ -z $used ]] && { log "${stage%%:*}: lineage of $WIN unknown, stopped"; break; }
    # a resume from model_M runs M .. M + k - 1: k more iterations on top of the "used" ones
    k=$(capped "$used" "${stage##*:}")
    ((k > 0)) || { log "${stage%%:*}: no iterations left under the cap ($used of $CAP behind $WIN)"; break; }
    name=robust_cont_${stage%%:*}
    task=${TASK_OF[$(dirname "$WIN")]:-$ROBUST}
    status "${stage%%:*}" "continue $WIN ($used it behind it) for $k it on $task"
    train "$name" "$task" "$k" resume "$(basename "$(dirname "$WIN")")" "$(basename "$WIN")"
    RC=$(run_dir "$name")
    [[ -z $RC ]] && break
    M=$(ckpt_iter "$WIN")
    BASE_OF[$RC]=$((used - M))  # its model_j (j >= M) has used + (j - M + 1) behind it
    TASK_OF[$RC]=$task
    EV=EV_R eval_run "$RC" 1 $(eval_points "$M" "$(last_ckpt "$RC")" 1000)
done

cosim_wait
python3 scripts/co_rl/sweeps/pick_best_blind.py --table "${EV_A[@]}" "${EV_B[@]}" "${EV_R[@]}" > "$Q/summary.txt" 2>&1
BEST=$(best_of "${EV_A[@]}" "${EV_B[@]}" "${EV_R[@]}")
{
    echo
    echo "best (15 cm, nominal + robust averaged where both exist): $BEST ($(cum_of "$BEST") training iterations behind it, cap $CAP)"
    echo "training iterations behind model_0 of each run (model_N: + N + 1):"
    for d in "${!BASE_OF[@]}"; do echo "  $d  ${BASE_OF[$d]}"; done
} >> "$Q/summary.txt"
status done "see $Q/summary.txt"
