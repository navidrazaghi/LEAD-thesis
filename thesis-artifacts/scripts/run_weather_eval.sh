#!/bin/bash
# Score the thesis models on the benchmark's adverse-weather routes.
#
# The proposal promised robustness under "rain, fog and night". What the thesis
# evaluated instead was parametric sensor degradation, and chapter six says so
# plainly. This closes that gap with the route set the benchmark already ships:
# forty routes whose weather is written into the route file itself -- thirty
# with heavy rain, twenty with heavy fog, twelve at night, thirty-four on wet
# road. None of them overlaps the thirty degradation routes, so the existing
# comparison is untouched by this one.
#
# The sensor condition is left nominal on purpose. The adversity here is the
# weather, and stacking synthetic damage on top would confound the two and
# answer neither question.
#
# The reference checkpoint runs last, so that stopping this early still leaves
# the three thesis models complete.
#
# It waits for the degradation campaign to finish rather than racing it: two
# CARLA servers on one GPU is how the earlier campaign lost cells.

set -u
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
PY=~/miniconda3/envs/lead/bin/python

# A gap in the process list is not proof the campaign finished; the log's own
# completion marker is.
echo "[$(date +%H:%M:%S)] waiting for the reference degradation run to finish"
while ! grep -q "^\[.*\] done$" ~/reference_eval.log 2>/dev/null; do
  sleep 300
done
echo "[$(date +%H:%M:%S)] it finished; settling before taking the GPU"
sleep 300
while pgrep -f "Carla[U]E4-Linux" > /dev/null 2>&1; do
  sleep 60
done

echo "[$(date +%H:%M:%S)] starting the weather evaluation"
$PY scripts/common/run_evaluation.py \
  --models rung0=outputs/rung0_baseline_post \
           rung2a=outputs/rung2a_curriculum_only_post \
           rung4=outputs/rung4_light_auxiliary_post \
           ref0=reference/seed0 \
  --routes src/lead/routes/eval_sets/weather.txt \
  --conditions none:0 \
  --out /home/razzaghi/LEAD/lead/results/weather_closed_loop.csv
echo "[$(date +%H:%M:%S)] weather evaluation done"
