#!/bin/bash
# Run the three simulator-free experiments now, then put the campaign back.
#
# They were chained after the campaign, which was the wrong order once the
# arithmetic was done: the campaign needs about seventy hours of the
# eighty-four available, so anything queued behind it might never run. These
# three take about an hour between them and are certain to finish, and one of
# them -- the intervention test -- is the highest-value experiment left, since
# it is what turns the thesis's central claim from a correlation into a causal
# one.
#
# So they go first. The campaign is resumable and loses nothing by waiting an
# hour; the experiments would have lost everything by waiting seventy.
#
# The cost measurement runs first of the three, while the card is quietest: a
# timing taken beside a running CARLA measures contention, not the model.

set -u
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
export OMP_NUM_THREADS=1

PY=~/miniconda3/envs/lead/bin/python
OUT=~/LEAD/lead/outputs
RES=~/LEAD/lead/results
SCRIPT=scripts/common/light_experiments.py

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_one() {
  local name=$1; shift
  log "=== $name ==="
  if $PY "$SCRIPT" "$@"; then
    log "$name finished"
  else
    log "FAIL $name exited with status $?"
  fi
}

log "starting the three experiments; the campaign is stopped"

run_one cost cost \
  --models "rung0=$OUT/rung0_baseline_post" \
           "rung2=$OUT/rung2_deformable_calibrated_post" \
           "rung4=$OUT/rung4_light_auxiliary_post" \
  --warmup 10 --repeats 50 --batch-size 8 \
  --out "$RES/cost.csv"

run_one intervention intervention \
  --models "rung2a=$OUT/rung2a_curriculum_only_post" \
           "rung2b=$OUT/rung2b_observability_ungated_post" \
           "rung3=$OUT/rung3_observability_gated_post" \
           "rung4=$OUT/rung4_light_auxiliary_post" \
  --batches 60 --batch-size 8 \
  --out "$RES/intervention.csv"

run_one ood ood \
  --models "rung0=$OUT/rung0_baseline_post" \
           "rung2a=$OUT/rung2a_curriculum_only_post" \
           "rung3=$OUT/rung3_observability_gated_post" \
           "rung4=$OUT/rung4_light_auxiliary_post" \
  --batches 60 --batch-size 8 \
  --out "$RES/ood.csv"

log "results:"
for f in cost intervention ood; do
  echo "--- $f ---"
  cat "$RES/$f.csv" 2>/dev/null || echo "  (missing)"
done

# Whatever happened above, the campaign goes back on. It is the long pole and
# every minute it is not running is a measurement not taken.
log "restarting the closed-loop campaign"
echo "
=== campaign resumed $(date +%H:%M:%S) after the light experiments ===" \
  >> ~/final_eval.log
nohup setsid bash ~/run_final_eval.sh >> ~/final_eval.log 2>&1 < /dev/null &
sleep 20
if pgrep -f "run_final_[e]val.sh" > /dev/null 2>&1; then
  log "campaign is running again"
else
  log "FAIL the campaign did not restart -- start it by hand"
fi
