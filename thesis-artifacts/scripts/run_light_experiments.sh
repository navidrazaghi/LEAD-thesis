#!/bin/bash
# The three simulator-free experiments, run once the GPU is free.
#
# All three need the GPU, and two of them are comparisons between models, so
# they must not run while the closed-loop campaign or its retry passes are
# using it. The cost measurement is the strictest: a timing taken while CARLA
# is running measures contention rather than the model. So this waits for
# everything else to finish before it starts, and runs the timing first, while
# the machine is quietest.
#
# What each one settles, in the order they run:
#
#   cost          whether the deformable operator is actually cheaper here. The
#                 thesis claims a linear rather than quadratic order and then
#                 declines to claim a speedup; this is the number that decides
#                 whether that caution was warranted.
#   intervention  whether the gated model relies less on a degraded modality,
#                 rather than merely weighting it less. This is the one the
#                 review called the highest-value experiment: it upgrades the
#                 central claim from a correlation to a causal one.
#   ood           whether robustness survives corruption outside the training
#                 family, which is the only test that separates robustness from
#                 in-distribution generalisation.
#
# Nothing here edits the thesis. Each experiment writes a CSV to be read and
# transcribed deliberately.

cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
export OMP_NUM_THREADS=1

PY=~/miniconda3/envs/lead/bin/python
OUT=~/LEAD/lead/outputs
RES=~/LEAD/lead/results
SCRIPT=scripts/common/light_experiments.py

R0="rung0=$OUT/rung0_baseline_post"
R2="rung2=$OUT/rung2_deformable_calibrated_post"
R2A="rung2a=$OUT/rung2a_curriculum_only_post"
R2B="rung2b=$OUT/rung2b_observability_ungated_post"
R3="rung3=$OUT/rung3_observability_gated_post"
R4="rung4=$OUT/rung4_light_auxiliary_post"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Brackets keep pgrep from matching this script's own command line.
busy() {
  pgrep -f "run_final_[e]val.sh" > /dev/null 2>&1 \
    || pgrep -f "retry_failed_[r]uns.sh" > /dev/null 2>&1 \
    || pgrep -f "run_[e]valuation.py" > /dev/null 2>&1 \
    || pgrep -f "Carla[U]E4" > /dev/null 2>&1
}

# The retry script polls on the same period, so the moment the campaign exits
# there is a window in which neither is running and this would start on top of
# the retry passes. Requiring the condition to hold twice, a settling period
# apart, closes it.
log "waiting for the closed-loop campaign and its retry passes"
while busy; do
  sleep 300
done
log "nothing running; confirming after a settling period"
sleep 600
while busy; do
  sleep 300
done
log "the GPU is free"

# A stale CARLA holding memory would distort the peak-memory figure.
free_gb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "GPU free memory: ${free_gb} MiB"

run_one() {
  local name=$1; shift
  log "=== $name ==="
  if $PY "$SCRIPT" "$@" 2>&1; then
    log "$name finished"
  else
    log "FAIL $name exited with status $?"
  fi
}

# Timing first, while nothing else has warmed the card.
run_one cost cost \
  --models "$R0" "$R2" "$R4" \
  --warmup 10 --repeats 50 --batch-size 8 \
  --out "$RES/cost.csv"

run_one intervention intervention \
  --models "$R2A" "$R2B" "$R3" "$R4" \
  --batches 60 --batch-size 8 \
  --out "$RES/intervention.csv"

run_one ood ood \
  --models "$R0" "$R2A" "$R3" "$R4" \
  --batches 60 --batch-size 8 \
  --out "$RES/ood.csv"

log "all three finished; results:"
for f in cost intervention ood; do
  echo "--- $f ---"
  cat "$RES/$f.csv" 2>/dev/null || echo "  (missing)"
done
