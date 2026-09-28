#!/bin/bash
# Both idle-card measurements in one window, then hand the GPU back.
#
# WHY BOTH AT ONCE
#
# The forward profile and the finer-grid pre-flight need the same scarce thing:
# a card nobody else is using. The first profile was taken under contention and
# the caveat written next to it claimed the shares were still trustworthy even
# though the milliseconds were not. That was wrong. Re-measured idle, the fusion
# blocks came out at 12.9% of the forward pass against the 2.2% recorded in
# docs/deformable_fusion.md -- contention had inflated the denominator far more
# than the part being measured, so the share was distorted about sixfold.
#
# Both measurements are therefore taken in the same window, on the same idle
# card, so the numbers in the documents come from one state of the machine
# rather than two.
#
# WHAT IT COSTS THE TRAINING IT INTERRUPTS
#
# Almost nothing, and that is checked rather than assumed. run_rung.sh passes
# resume_from_last_checkpoint=true, so a stopped rung restarts from its last
# saved epoch; the loss is the partial epoch in flight. The alternative is
# waiting about thirty hours for the dense line to finish, which buys a slightly
# cheaper interruption at the price of the measurement not existing.
#
# The caller stops the training. This script only takes the lock, measures, and
# starts the dense line again -- which resumes rather than restarts.
#
# Everything runs inside a function called on the last line, so editing this
# file while it runs cannot corrupt the running program.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local results="$HOME/LEAD/lead/results"
  local model="outputs/rung0_baseline_post"
  local python="$HOME/miniconda3/envs/lead/bin/python"
  local env="LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1"

  if [ ! -f "$model/model_0009.pth" ]; then
    echo "[$(date +%H:%M:%S)] no dense checkpoint at $model"
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 172800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  # The lock is released when its holder exits, which is a moment before that
  # job's dataloader workers are reaped. Timing against their teardown would
  # reintroduce the contention this exists to avoid.
  sleep 90
  echo "[$(date +%H:%M:%S)] card idle: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"

  mkdir -p "$results"

  echo "[$(date +%H:%M:%S)] forward profile, both operators, idle card"
  env $env "$python" scripts/common/forward_profile.py \
    --models rung0=outputs/rung0_baseline_post \
             rung4=outputs/rung4_light_auxiliary_post \
    > "$results/forward_profile.txt" 2>&1
  echo "[$(date +%H:%M:%S)] profile exited $?"

  echo "[$(date +%H:%M:%S)] fusion cost by difference, both operators"
  env $env "$python" scripts/common/fusion_cost_by_difference.py \
    --models rung0=outputs/rung0_baseline_post \
             rung4=outputs/rung4_light_auxiliary_post \
    > "$results/fusion_cost_by_difference.txt" 2>&1
  echo "[$(date +%H:%M:%S)] difference measurement exited $?"

  echo "[$(date +%H:%M:%S)] finer-grid pre-flight"
  env $env "$python" scripts/common/finer_grid_preflight.py \
    --model "$model" --anchor-stride 16 --compile \
    > "$results/finer_grid_preflight.txt" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] pre-flight exited $code"
  # Exit 1 here is a result, not a malfunction: the script refuses to conclude
  # from timings that cannot be right, and that refusal is the thing worth
  # knowing. Either way the dense line gets its card back.
  if [ "$code" -ne 0 ]; then
    echo "--- why it stopped ---"
    tail -6 "$results/finer_grid_preflight.txt"
  fi

  exec 9>&-
  echo "[$(date +%H:%M:%S)] handing the card back to the dense line"
  exec bash scripts/common/run_dense_line.sh
}

main "$@"
