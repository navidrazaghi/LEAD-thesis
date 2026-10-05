#!/bin/bash
# Relaunch of the degradation pilot. ulimit and TIMM_USE_OLD_CACHE are the two
# settings that fail quietly rather than loudly if they are missing.
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
exec ~/miniconda3/envs/lead/bin/python scripts/common/run_evaluation.py \
  --models rung0=outputs/rung0_baseline_post rung3=outputs/rung3_observability_gated_post \
  --routes src/lead/routes/eval_sets/degradation_pilot.txt \
  --conditions none:0 camera:0.5 camera:1.0 lidar:0.5 lidar:1.0 \
  --out results/pilot.csv
