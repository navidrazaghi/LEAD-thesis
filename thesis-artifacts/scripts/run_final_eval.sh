#!/bin/bash
# The closed-loop campaign the thesis claim rests on.
#
# Open-loop waypoint error is an average over frames, and driving fails in rare
# moments rather than average ones, so it can only triage which models are worth
# the expensive budget. This is where the claim is actually settled.
#
# Three models, chosen so the comparison answers one question:
#   rung0   the untouched baseline
#   rung2a  the degradation curriculum alone -- currently the best model in the
#           open-loop table, and the rival explanation for everything
#   the better of rung3 / rung4, i.e. the gate on top of that curriculum
# Read against rung2a, the third model is the gate and nothing else.
#
# Three conditions. camera:0.5 is dropped because it was measured to do no
# damage at all, and its budget buys routes instead -- with a per-route driving
# score spread of ~31, route count is the only real lever on what is detectable.

cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1

PY=~/miniconda3/envs/lead/bin/python
OUT=~/LEAD/lead/outputs
RES=~/LEAD/lead/results

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Brackets keep pgrep from matching this script's own command line.
log "waiting for training and the open-loop analysis to finish"
while pgrep -f "run_ablation_[c]hain" > /dev/null 2>&1 \
   || pgrep -f "after_[t]raining.sh" > /dev/null 2>&1; do
  sleep 120
done
log "both are done"

# Pick the gated candidate on its clean open-loop error. This is triage, which
# is what open-loop is good for -- it decides who gets measured, not who wins.
THIRD=""
if [ -f "$RES/mechanism.csv" ]; then
  THIRD=$($PY - <<'PYEOF'
import csv
import pathlib

path = pathlib.Path.home() / "LEAD/lead/results/mechanism.csv"
clean = {}
for row in csv.DictReader(path.open()):
    if row["condition"] != "none:0":
        continue
    value = row.get("waypoint_l2") or ""
    if value not in ("", "None"):
        clean[row["model"]] = float(value)
ranked = [(clean[m], m) for m in ("rung3", "rung4") if m in clean]
print(min(ranked)[1] if ranked else "")
PYEOF
)
fi
[ -z "$THIRD" ] && THIRD=rung3
case "$THIRD" in
  rung4) THIRD_DIR=rung4_light_auxiliary_post ;;
  *)     THIRD_DIR=rung3_observability_gated_post ;;
esac
log "gated candidate: $THIRD ($THIRD_DIR)"

for d in rung0_baseline_post rung2a_curriculum_only_post "$THIRD_DIR"; do
  if [ ! -f "$OUT/$d/model_0009.pth" ]; then
    log "FAIL missing checkpoint: $d"
    exit 1
  fi
done

log "starting the sweep: 3 models x 3 conditions x 30 routes = 270 runs"
log "resumable, so an interruption costs one route rather than the campaign"
# Condition order matters now that the models run innermost: the sweep finishes
# one condition at a time, so a campaign cut short still has whole conditions.
# none:0 first because every comparison is read against it, then lidar, which is
# where the open-loop gap between the models was widest.
$PY scripts/common/run_evaluation.py \
  --models \
    rung0=outputs/rung0_baseline_post \
    rung2a=outputs/rung2a_curriculum_only_post \
    "$THIRD=outputs/$THIRD_DIR" \
  --routes src/lead/routes/eval_sets/degradation_30.txt \
  --conditions none:0 lidar:1.0 camera:1.0 \
  --out "$RES/closed_loop.csv"
status=$?

log "sweep exited with status $status; writing the summary"
$PY scripts/common/summarize_closed_loop.py \
  --csv "$RES/closed_loop.csv" --reference rung2a \
  > "$RES/closed_loop_table.txt" 2>&1
cat "$RES/closed_loop_table.txt"
exit $status
