#!/bin/bash
# Sweep v3 (stair_step_lift_v3.yaml) continued ONE trial at a time.
#
# The GB10 GPU time-slices parallel trainings: the total throughput stays ~30k env-steps/s, so four
# parallel trials each get a quarter (12.8 s/it) and all finish together at the end. Run one at a
# time, the best trial finishes first and the others can be dropped after looking at it.
#
# Each trial resumes from its newest checkpoint (any *_sweep00<t> run dir of the experiment, so the
# script can itself be stopped and restarted) and trains up to model_4999. Same task, overrides and
# 4096 envs as the parallel sweep; a trial that already reached the target is skipped.
#
#   systemd-run --user --unit=flamingo-stairstep-v3-seq --working-directory=$PWD \
#     /bin/bash scripts/co_rl/sweeps/stair_step_lift_v3_sequential.sh 3 1 2 0
#
# Arguments: trial order (default 3 1 2 0). Logs: <sweep dir>/sweep00<t>_seq.log.
set -u
cd "$(dirname "$0")/../../.."

R=logs/co_rl/Flamingo_Light_Rough_Stair_Step_Lift/ppo
SW=$R/_sweeps/stair_step_lift_v3_2026-10-27_13-40-57
TARGET=5000
ORDER=${*:-3 1 2 0}

for t in $ORDER; do
  task=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-v1-ppo
  [ "$t" -ge 2 ] && task=Isaac-Velocity-Rough-Flamingo-Light-Stair-Step-Lift-Perceptive-v1-ppo

  # newest checkpoint over every run dir of this trial
  ckpt=""; ckpt_it=-1
  for f in $R/*_sweep00$t/model_*.pt; do
    [ -f "$f" ] || continue
    it=$(basename "$f" .pt); it=${it#model_}
    if [ "$it" -gt "$ckpt_it" ]; then ckpt_it=$it; ckpt=$f; fi
  done
  if [ -z "$ckpt" ]; then
    echo "[seq] trial $t: no checkpoint found, skipped"; continue
  fi
  # learn(n) runs n iterations after the loaded one, so TARGET - it ends at model_(TARGET-1)
  remaining=$((TARGET - ckpt_it))
  if [ "$remaining" -le 1 ]; then
    echo "[seq] trial $t: already at $ckpt, skipped"; continue
  fi

  echo "[seq] $(date '+%F %T') trial $t: $task from $ckpt, $remaining iterations"
  ./run_gb10.sh -u scripts/co_rl/train.py --task "$task" --algo ppo --max_iterations "$remaining" \
    --run_name "sweep00$t" --param_overrides "$SW/sweep00${t}_overrides.json" \
    --resume True --load_run "$(basename "$(dirname "$ckpt")")" --checkpoint "$(basename "$ckpt")" \
    --headless --num_envs 4096 --batch_exit >> "$SW/sweep00${t}_seq.log" 2>&1
  echo "[seq] $(date '+%F %T') trial $t: exit $?"
done
echo "[seq] $(date '+%F %T') done"
