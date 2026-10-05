#!/bin/bash
# Re-measure the forward pass with nothing else on the card.
#
# The first measurement is void. The retry watcher fired while it was running
# -- it saw no campaign process, concluded the sweep had finished, and started
# its own CARLA -- so the timings were taken against a busy GPU. That matters
# here and nowhere else: the accuracy experiments compute the same arithmetic
# whatever else is running, but a timing under contention measures the
# contention.
#
# It also matters in the direction that would have flattered the conclusion.
# The first run put the deformable operator about ten percent slower than dense
# attention; if CARLA came up between the first model and the others, some of
# that gap is the neighbour rather than the operator. A number that happens to
# support the argument is exactly the one to distrust.
#
# Three minutes of campaign time. The campaign is resumable and loses nothing.

set -u
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
export OMP_NUM_THREADS=1

PY=~/miniconda3/envs/lead/bin/python
OUT=~/LEAD/lead/outputs
RES=~/LEAD/lead/results

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "stopping the campaign for the re-measurement"
pkill -f "run_final_[e]val.sh"
pkill -f "run_[e]valuation.py"
sleep 5
pkill -f "Carla[U]E4-Linux"
sleep 10
pkill -9 -f "Carla[U]E4-Linux" 2>/dev/null
sleep 5

# Anything still holding memory would distort the peak-memory column.
remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
log "processes still on the GPU: $remaining"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

log "measuring"
$PY scripts/common/light_experiments.py cost \
  --models "rung0=$OUT/rung0_baseline_post" \
           "rung2=$OUT/rung2_deformable_calibrated_post" \
           "rung4=$OUT/rung4_light_auxiliary_post" \
  --warmup 20 --repeats 100 --batch-size 8 \
  --out "$RES/cost.csv"
log "cost re-measured with status $?"
cat "$RES/cost.csv"

log "restarting the campaign"
echo "
=== campaign resumed $(date +%H:%M:%S) after re-measuring cost ===" \
  >> ~/final_eval.log
nohup setsid bash ~/run_final_eval.sh >> ~/final_eval.log 2>&1 < /dev/null &
sleep 20
if pgrep -f "run_final_[e]val.sh" > /dev/null 2>&1; then
  log "campaign is running again"
else
  log "FAIL the campaign did not restart -- start it by hand"
fi
