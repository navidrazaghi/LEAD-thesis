#!/bin/bash
# Score rung2d on the axis it was trained for, and on the one everything else uses.
#
# rung2d carries the deployment perturbation families -- a blocked lens and a
# drifting ego state -- on top of the appearance curriculum. It finished
# training and sat unevaluated, and until this week it could not have been
# evaluated properly: the inference config exposed a modality and a severity and
# nothing else, so the families were reachable from training only.
#
# Two axes are scored here for that reason.
#
# The deployment conditions are the question the rung was built to answer. They
# are also the regime the caution governor never had: a blocked lens or a slow
# speedometer is a fault the car still drives through, unlike the destroyed
# sensors where nothing drives at all.
#
# The sensor conditions are the ladder's own protocol, and rung2a -- this rung's
# control -- already has them in results/closed_loop.csv. Without them there is
# no comparison, only a number.
#
# Separate results file: closed_loop.csv is the thesis's and the thesis is
# finished. The harness keeps and skips rows already present, so a stopped run
# resumes by being started again.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local model="outputs/rung2d_deployment_families_post"
  local out="$HOME/LEAD/lead/results/closed_loop_rung2d.csv"
  local routes="src/lead/routes/eval_sets/degradation_30.txt"

  if [ ! -f "$model/model_0009.pth" ]; then
    echo "[$(date +%H:%M:%S)] $model has no final checkpoint"
    exit 1
  fi

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 604800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  echo "[$(date +%H:%M:%S)] scoring 30 routes x 5 conditions"
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
    "$HOME/miniconda3/envs/lead/bin/python" scripts/common/run_evaluation.py \
      --models "rung2d=$model" \
      --routes "$routes" \
      --conditions none:0 occlusion:1.0 ego_state:1.0 lidar:1.0 camera:1.0 \
      --out "$out"
  local code=$?
  echo "[$(date +%H:%M:%S)] evaluation exited $code"
  echo "[$(date +%H:%M:%S)] rows: $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 ))"
  exit "$code"
}

main "$@"
