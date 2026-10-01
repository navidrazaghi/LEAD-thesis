#!/bin/bash
# Score the retrained baseline, on the protocol every other rung was scored on.
#
# WHY THIS RUN MATTERS MORE THAN A USUAL EVALUATION
#
# The baseline the thesis compares everything against never converged: its
# training loss bottomed at epoch two and ended 2.5x above its best. Retrained
# on LEAD's own recipe -- effective batch 64 through accumulation, 31 epochs, the
# shipped learning rate untouched -- the curve descends monotonically to its
# minimum at the last epoch and ends 5.4x lower than before.
#
# So there is now a baseline that trained properly, and no number for it. Every
# comparison in the thesis is measured against the old one, including the 25.0
# intact driving score that anchors the ablation ladder.
#
# The direction of the correction is not known and is not assumed to be
# favourable. A baseline that trains properly should score higher, which narrows
# the margin every other rung is reported to have over it. Some reported
# advantages may shrink; some may not survive.
#
# THE PROTOCOL IS COPIED, NOT CHOSEN
#
# The same 30 routes of degradation_30.txt under intact, LiDAR destroyed and
# camera destroyed. Ninety runs. Any deviation would make the new number
# incomparable with the table it is meant to correct, which would defeat the
# purpose of measuring it.
#
# ORDERING
#
# This waits for the posttrain's own final checkpoint rather than for the lock,
# because flock promises exclusion and not order. Waiting on the lock alone
# would let this start against a half-trained checkpoint if the training driver
# released between its two stages.
#
# Results go to their own file. closed_loop.csv is the thesis's and is cited in a
# submitted document; it should not grow.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local model="outputs/rung0_lead_recipe_post"
  local final="model_0009.pth"
  local out="$HOME/LEAD/lead/results/closed_loop_new_baseline.csv"
  local routes="src/lead/routes/eval_sets/degradation_30.txt"

  echo "[$(date +%H:%M:%S)] waiting for the posttrain to finish"
  while [ ! -f "$model/$final" ]; do
    if ! ps -eo args --no-headers \
         | grep -qE "^bash (.*/)?scripts/common/run_baseline_lead_recipe[.]sh$"; then
      echo "[$(date +%H:%M:%S)] the training driver is gone and $final never appeared"
      exit 1
    fi
    sleep 600
  done
  echo "[$(date +%H:%M:%S)] checkpoint present"

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 604800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  # The harness keeps and skips rows already present, so an interruption is
  # resumed by starting this again rather than by remembering anything.
  echo "[$(date +%H:%M:%S)] scoring 30 routes x 3 conditions"
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
    "$HOME/miniconda3/envs/lead/bin/python" scripts/common/run_evaluation.py \
      --models "rung0_lead=$model" \
      --routes "$routes" \
      --conditions none:0 lidar:1.0 camera:1.0 \
      --out "$out"
  local code=$?
  echo "[$(date +%H:%M:%S)] evaluation exited $code"
  echo "[$(date +%H:%M:%S)] rows: $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) of 90"
  echo "  Compare against rung0's rows in results/closed_loop.csv: same routes,"
  echo "  same conditions, a baseline that converged against one that did not."
  exit "$code"
}

main "$@"
