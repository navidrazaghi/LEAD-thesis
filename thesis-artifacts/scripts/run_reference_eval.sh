#!/bin/bash
# Score the published reference checkpoint under this thesis's own protocol.
#
# Same thirty routes, same three sensor conditions, same harness, and the same
# damage seed derived from each route name. The only thing that differs from
# the thesis's own rows is the checkpoint: identical architecture and parameter
# count, trained by its authors on the full dataset for thirty epochs instead
# of on a 450-log subset for nine. So whatever gap the numbers show is a gap in
# data and training, not in design.
#
# Resumable: rows already present in the CSV are skipped, so an interruption
# costs only the route it was on. The harness restarts CARLA every eight routes
# and abandons a cell that has failed twice, exactly as it did for the thesis's
# own campaign.
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1
export LEAD_RUNTIME_TYPE_CHECKING=false
PY=~/miniconda3/envs/lead/bin/python

echo "[$(date +%H:%M:%S)] starting reference evaluation"
$PY scripts/common/run_evaluation.py \
  --models ref0=reference/seed0 \
  --routes src/lead/routes/eval_sets/degradation_30.txt \
  --conditions none:0 lidar:1.0 camera:1.0 \
  --out /home/razzaghi/LEAD/lead/results/reference_closed_loop.csv
echo "[$(date +%H:%M:%S)] done"
