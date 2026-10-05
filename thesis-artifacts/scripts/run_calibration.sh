#!/bin/bash
# Let the caution governor's scalar converge on the calibration routes.
#
# Twenty clear-weather routes the scored sets do not use, so the value this
# produces is not fitted to the routes the governor is later scored on. That
# disjointness is enforced by the route selector and checked there; this script
# only has to use the right file.
#
# The condition is joint degradation at full severity, which is the only regime
# the governor acts in -- measured, not assumed: under intact sensors and under
# either single-modality family the caution signal sits at or near zero, so a
# calibration run there would see no risk events, let the scalar decay to zero
# and produce a number that means "never act".
#
# A fresh agent is built per route, so the scalar starts from
# caution_initial_lambda every time and adapts within that route alone. What
# comes out is therefore one converged value per route, appended to
# caution_calibration.jsonl by the agent's destroy hook.
#
# The ulimit is not optional: the default 1024 starves the LMDB cache readers.

set -u
cd "$HOME/LEAD/lead" || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false

MODEL=outputs/rung4_light_auxiliary_post_ensemble
OUT="$HOME/LEAD/lead/results/calibration"
mkdir -p "$OUT"

if [ ! -d "$MODEL" ]; then
  echo "no ensemble checkpoint at $MODEL" >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] calibrating the governor on the calibration routes"

~/miniconda3/envs/lead/bin/python scripts/common/run_evaluation.py \
  --models rung4_ens="$MODEL" \
  --routes src/lead/routes/eval_sets/calibration.txt \
  --conditions both:1.0 \
  --out results/calibration_closed_loop.csv \
  --config evaluation.inference.use_caution_governor=true \
           evaluation.inference.caution_signal=ensemble \
           evaluation.save_path="$OUT" \
  || { echo "calibration run failed"; exit 1; }

echo "[$(date +%H:%M:%S)] calibration done"
