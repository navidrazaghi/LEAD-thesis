#!/bin/bash
# Closed-loop evaluation of the dense control, on the thesis protocol.
#
# WHAT IT IS FOR
#
# rung2a_dense_curriculum is the control the dense line needed: the degradation
# curriculum on the dense operator and nothing else. Against rung2a it isolates
# the operator under the curriculum; against rung0 it re-measures the curriculum
# on the dense path, which answers something the thesis could not -- whether its
# one solid positive result ever depended on the sparse operator at all.
#
# Neither comparison means anything unless this is scored the same way they
# were, so the protocol is copied from what produced results/closed_loop.csv:
# the 30 routes of degradation_30.txt under intact, LiDAR destroyed and camera
# destroyed. Ninety routes.
#
# WHY A SEPARATE RESULTS FILE
#
# results/closed_loop.csv is the thesis's, and the thesis is finished. The
# harness would append to it harmlessly -- it keeps and skips existing rows --
# but a file that is cited in a submitted document should not be growing. The
# analysis merges the two.
#
# The lock is the same one the training drivers take. CARLA and a trainer on one
# card would make both meaningless.
#
# Everything runs inside a function called on the last line: bash reads a script
# incrementally, so editing this file while it runs would corrupt the running
# program.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local model="outputs/rung2a_dense_curriculum_post"
  local out="$HOME/LEAD/lead/results/closed_loop_dense.csv"
  local routes="src/lead/routes/eval_sets/degradation_30.txt"

  if [ ! -f "$model/model_0009.pth" ]; then
    echo "[$(date +%H:%M:%S)] $model has no final checkpoint; nothing to score"
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 172800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  # The harness keeps and skips rows already in the CSV, so an interrupted run
  # resumes by being started again rather than by remembering anything.
  echo "[$(date +%H:%M:%S)] scoring 30 routes x 3 conditions"
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
    "$HOME/miniconda3/envs/lead/bin/python" scripts/common/run_evaluation.py \
      --models "rung2a_dense=$model" \
      --routes "$routes" \
      --conditions none:0 lidar:1.0 camera:1.0 \
      --out "$out"
  local code=$?
  echo "[$(date +%H:%M:%S)] evaluation exited $code"
  echo "[$(date +%H:%M:%S)] rows now in $(basename "$out"): $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))"
  exit "$code"
}

main "$@"
