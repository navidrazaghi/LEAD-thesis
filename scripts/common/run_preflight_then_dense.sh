#!/bin/bash
# Run the finer-grid pre-flight in the gap, then hand the GPU to the dense line.
#
# WHY A WRAPPER RATHER THAN JUST WAITING
#
# The pre-flight needs an idle card. Its verdict rests on one part's share of
# the whole forward pass, and a training run sharing the GPU does not inflate
# every part equally -- the last profile taken under contention came out about
# fourfold high with one module's share visibly wrong. Reading it off a busy
# card would produce a number that looks like an answer and is not one.
#
# But the card is booked for roughly two days: rung2d's posttrain now, the dense
# line behind it. The only idle moment is the handover between them, which lasts
# as long as nobody takes it.
#
# Waiting on the lock alongside the dense line would be a coin toss -- flock
# makes no ordering promise, so the pre-flight could lose and wait out another
# thirty-five hours. This takes the lock itself, runs the few minutes it needs,
# releases, and only then starts the dense line. The ordering is structural
# rather than lucky.
#
# Everything runs inside a function called on the last line, so editing this
# file while it waits cannot corrupt the running program -- the trap that cost
# rung2d its first posttrain.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local out="$HOME/LEAD/lead/results/finer_grid_preflight.txt"
  local model="outputs/rung0_baseline_post"
  local python="$HOME/miniconda3/envs/lead/bin/python"

  if [ ! -f "$model/model_0009.pth" ]; then
    echo "[$(date +%H:%M:%S)] no dense checkpoint at $model; nothing to profile"
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 172800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired; the card is ours"

  # Let the previous job's workers actually exit before timing anything. The
  # lock is released when its holder does, which is a moment before its dataloader
  # workers are reaped.
  sleep 60

  mkdir -p "$(dirname "$out")"
  echo "[$(date +%H:%M:%S)] running the pre-flight"
  # LEAD_RUNTIME_TYPE_CHECKING must be off under --compile: beartype and Dynamo
  # cannot run together.
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
    "$python" scripts/common/finer_grid_preflight.py \
      --model "$model" \
      --anchor-stride 16 \
      --compile \
    > "$out" 2>&1
  local code=$?
  echo "[$(date +%H:%M:%S)] pre-flight exited $code; output in $out"
  if [ "$code" -ne 0 ]; then
    echo "--- last lines ---"
    tail -20 "$out"
  fi

  # Release before handing over: run_dense_line.sh takes the same lock, and
  # holding it here would make it wait on this process forever.
  exec 9>&-
  echo "[$(date +%H:%M:%S)] handing the card to the dense line"
  exec bash scripts/common/run_dense_line.sh
}

main "$@"
