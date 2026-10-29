#!/bin/bash
# Two-stage SYMMETRIC stair-step training from the v01_2_2 flat stand-drive.
#
# Every policy that lifts its wheels onto risers above the axle descends from the sweep-v2 policies,
# which learned on the loose step-up rule (a wheel rolled or bounced over a riser counted) and were
# then moved to the strict rule (v3: the leg has to lift the wheel). All of them start a staircase
# with the right wheel. A strict-rule run straight from the stand-drive (with the mirror loss) never
# lifted a wheel in 10000 it: nothing pays before the first real lift. So, with the mirror loss in
# both stages:
#   stage 1  Stair-Step-Mirror-Perceptive (v2 rules, risers 5-15 cm, v2 trial-2 ZMP shaping x3) from
#            the stand-drive, STAGE1_IT iterations; extended once by STAGE1_EXTRA if the last 200
#            iterations climb fewer than MIN_RISERS risers per episode (v2 took off at ~1750-1950 it)
#   stage 2  Stair-Step-Lift-Mirror-Gain-Perceptive (strict rule, risers 9-15 cm, potential retraction
#            reward) from the last stage-1 checkpoint, STAGE2_IT iterations
#
#   systemd-run --user --unit=flamingo-two-stage --working-directory=$PWD \
#     /bin/bash scripts/co_rl/sweeps/stair_step_two_stage_mirror.sh
set -u
cd "$(dirname "$0")/../../.."

RUN=${RUN:-two_stage}
STAGE1_IT=${STAGE1_IT:-3000}
STAGE1_EXTRA=${STAGE1_EXTRA:-2000}
STAGE2_IT=${STAGE2_IT:-5000}
MIN_RISERS=${MIN_RISERS:-0.5}
NUM_ENVS=${NUM_ENVS:-4096}
SD=logs/co_rl/Flamingo_Light_v01_2_2_Flat_Stand_Drive/ppo/2026-10-23_11-56-03_v0122_stand_drive/model_1499.pt
PY=/home/midbar/ws/isaaclab/env_isaaclab/bin/python
S1_ROOT=logs/co_rl/Flamingo_Light_Rough_Stair_Step_Mirror/ppo
LOG=logs/co_rl/_runlogs/${RUN}.log
mkdir -p logs/co_rl/_runlogs
OV=logs/co_rl/_runlogs/${RUN}_stage1_overrides.json
echo '{"env.rewards.zmp_tracking.weight": 3.0, "env.rewards.cop_transfer.weight": 3.0}' > "$OV"

say() { echo "[two-stage] $(date '+%F %T') $*" | tee -a "$LOG"; }

latest_ckpt() {  # newest model_*.pt over every run dir of the given root/suffix
  local best="" best_it=-1 f it
  for f in "$1"/*_"$2"/model_*.pt; do
    [ -f "$f" ] || continue
    it=$(basename "$f" .pt); it=${it#model_}
    if [ "$it" -gt "$best_it" ]; then best_it=$it; best=$f; fi
  done
  echo "$best"
}

risers_tail() {  # mean Curriculum/gait/risers_climbed of the last 200 logged iterations of a run dir
  "$PY" - "$1" <<'EOF'
import sys
from tensorboard.backend.event_processing import event_accumulator as ea
a = ea.EventAccumulator(sys.argv[1], size_guidance={ea.SCALARS: 0})
a.Reload()
ev = a.Scalars("Curriculum/gait/risers_climbed")[-200:]
print(f"{sum(e.value for e in ev) / max(len(ev), 1):.3f}")
EOF
}

# ---- stage 1: loose rule + mirror, from the stand-drive
if [ -z "$(latest_ckpt "$S1_ROOT" "${RUN}_s1")" ]; then
  say "stage 1: $STAGE1_IT it from $SD"
  ./run_gb10.sh -u scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Mirror-Perceptive-v1-ppo \
    --algo ppo --headless --num_envs "$NUM_ENVS" --max_iterations "$STAGE1_IT" --batch_exit --run_name "${RUN}_s1" \
    --param_overrides "$OV" --warmstart_ckpt "$SD" --warmstart_widen --warmstart_std_scale 1.5 >> "$LOG" 2>&1
fi
S1=$(latest_ckpt "$S1_ROOT" "${RUN}_s1")
S1_DIR=$(dirname "$S1")
R1=$(risers_tail "$S1_DIR")
say "stage 1 done: $S1, risers/episode (last 200 it) $R1"
if "$PY" -c "import sys; sys.exit(0 if float('$R1') < float('$MIN_RISERS') else 1)"; then
  it=$(basename "$S1" .pt); it=${it#model_}
  say "stage 1 below $MIN_RISERS risers: extending by $STAGE1_EXTRA it from $S1"
  ./run_gb10.sh -u scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Mirror-Perceptive-v1-ppo \
    --algo ppo --headless --num_envs "$NUM_ENVS" --max_iterations "$STAGE1_EXTRA" --batch_exit --run_name "${RUN}_s1" \
    --param_overrides "$OV" --resume True --load_run "$(basename "$S1_DIR")" --checkpoint "$(basename "$S1")" >> "$LOG" 2>&1
  S1=$(latest_ckpt "$S1_ROOT" "${RUN}_s1")
  R1=$(risers_tail "$(dirname "$S1")")
  say "stage 1 extended: $S1, risers/episode (last 200 it) $R1"
fi

# ---- stage 2: strict rule + mirror + potential retraction, from the last stage-1 checkpoint
say "stage 2: $STAGE2_IT it from $S1"
./run_gb10.sh -u scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Mirror-Gain-Perceptive-v1-ppo \
  --algo ppo --headless --num_envs "$NUM_ENVS" --max_iterations "$STAGE2_IT" --batch_exit --run_name "${RUN}_s2" \
  --warmstart_ckpt "$S1" --warmstart_widen --warmstart_std_scale 1.5 >> "$LOG" 2>&1
say "stage 2 done: exit $?"
