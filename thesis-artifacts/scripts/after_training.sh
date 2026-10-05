#!/bin/bash
# Wait for the ablation chain to finish, then run every analysis that does not
# need CARLA, over the whole ladder.
#
# This lives on the server on purpose. Driving it from a laptop session would
# tie a twenty-hour wait to that session surviving twenty hours, and the point
# of the wait is that nobody has to sit through it.

cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false

PY=~/miniconda3/envs/lead/bin/python
OUT=~/LEAD/lead/outputs
RES=~/LEAD/lead/results
STAMP=$(date +%Y%m%d_%H%M)

log() { echo "[$(date +%H:%M:%S)] $*"; }

# The bracket keeps pgrep from matching this script's own command line.
log "waiting for the ablation chain to finish"
while pgrep -f "run_ablation_[c]hain" > /dev/null 2>&1; do
  sleep 120
done
log "chain process is gone"

# A chain that stopped early leaves the ladder with holes, and a mechanism
# table missing its control rows is worse than no table: it invites exactly the
# attribution the extra rungs exist to rule out. So check, and say so.
MODELS=()
MISSING=()
for pair in \
  "rung0=rung0_baseline_post" \
  "rung1=rung1_deformable_free_post" \
  "rung2=rung2_deformable_calibrated_post" \
  "rung2a=rung2a_curriculum_only_post" \
  "rung2b=rung2b_observability_ungated_post" \
  "rung3=rung3_observability_gated_post"   "rung4=rung4_light_auxiliary_post"; do
  name=${pair%%=*}
  dir=${pair#*=}
  if [ -f "$OUT/$dir/model_0009.pth" ]; then
    MODELS+=("$name=outputs/$dir")
  else
    MISSING+=("$name")
  fi
done

log "models found: ${MODELS[*]}"
if [ ${#MISSING[@]} -gt 0 ]; then
  log "WARNING missing checkpoints: ${MISSING[*]} -- the table below has holes"
fi
if [ ${#MODELS[@]} -lt 2 ]; then
  log "too few models to compare; stopping"
  exit 1
fi

log "running the mechanism probe over ${#MODELS[@]} models"
$PY scripts/common/analyze_gate.py \
  --models "${MODELS[@]}" \
  --conditions none:0 camera:0.5 camera:1.0 lidar:0.5 lidar:1.0 \
  --batches 60 --batch-size 8 --workers 4 \
  --out "$RES/mechanism.csv" || { log "FAIL mechanism probe"; exit 1; }

log "writing the summary table"
$PY scripts/common/summarize_mechanism.py --csv "$RES/mechanism.csv" \
  > "$RES/mechanism_table_$STAMP.txt" 2>&1
cp "$RES/mechanism_table_$STAMP.txt" "$RES/mechanism_table_latest.txt"

# Two figures, not one. rung2b carries the observability head but no gate, so
# its last column reads "no gate" beside an otherwise identical panel -- which
# is the control the gated figure needs to be read against.
for pair in "rung3=rung3_observability_gated_post" "rung2b=rung2b_observability_ungated_post" "rung4=rung4_light_auxiliary_post"; do
  name=${pair%%=*}
  dir=${pair#*=}
  [ -f "$OUT/$dir/model_0009.pth" ] || continue
  log "drawing the figure for $name"
  $PY scripts/common/plot_observability.py \
    --model "$OUT/$dir" \
    --out "$RES/observability_${name}.png" || log "FAIL figure for $name"
done

log "done. results:"
ls -l "$RES"/mechanism.csv "$RES"/mechanism_table_latest.txt "$RES"/observability_*.png 2>/dev/null
echo
cat "$RES/mechanism_table_latest.txt"
