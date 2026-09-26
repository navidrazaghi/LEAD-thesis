#!/bin/bash
# Fit the waypoint ensemble on top of a finished rung, with that rung frozen.
#
# Usage: run_ensemble_finetune.sh <source-rung-output-dir> [config overrides...]
#
# The ensemble measures what the planning context leaves undetermined. That only
# means anything if the context is fixed: a readout whose inputs move while it
# is learning them is measuring the movement as much as the scene. So everything
# but the ensemble is frozen, which also makes this cheap -- about three million
# trainable parameters against seventy-two million, and no gradient through the
# backbone at all.
#
# Freezing rather than a small learning rate is deliberate. A small rate still
# moves the features, just slowly, and the members would be fitted to a moving
# target that each of them sees at a slightly different moment.
#
# Two settings are not free choices:
#
#   resume_from_last_checkpoint=false makes load_state_dict non-strict, which is
#   required here. The source checkpoint has no ensemble in it, so a strict load
#   refuses the very weights this run exists to add. It is the same reason the
#   first launch of a posttrain sets it false.
#
#   use_planning_decoder=true, because the context the members read is the one
#   that decoder builds; the model refuses to construct otherwise.
#
# The ulimit is not optional: the default 1024 starves the LMDB cache readers
# and training crawls at a fiftieth of the speed without ever failing outright.

set -u

if [ $# -lt 1 ]; then
  echo "usage: $0 <source-rung-output-dir> [overrides...]" >&2
  exit 2
fi

SOURCE=$1; shift
NAME=$(basename "$SOURCE")_ensemble

cd "$HOME/LEAD/lead" || exit 1
ulimit -n 65536

# The newest model_*.pth of the source rung is its final epoch.
WEIGHTS=$(ls -t "$SOURCE"/model_*.pth 2>/dev/null | head -n1 || true)
if [ -z "$WEIGHTS" ]; then
  echo "no model_*.pth under $SOURCE" >&2
  exit 1
fi
echo "[$(date +%H:%M:%S)] fitting the ensemble on $WEIGHTS"

exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue WANDB_MODE=offline \
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
  LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}" \
  "$HOME/miniconda3/envs/lead/bin/python" -m lead.training.train \
  training.data.read_from_cache_store=true \
  training.optimization.batch_size=8 \
  training.optimization.num_epochs=3 \
  training.experiment.initial_weights_file="$WEIGHTS" \
  training.experiment.resume_from_last_checkpoint=false \
  training.experiment.freeze_except="[waypoint_ensemble]" \
  training.experiment.output_dir="$HOME/LEAD/lead/outputs/$NAME" \
  policy.transfuser.use_planning_decoder=true \
  policy.transfuser.use_waypoint_ensemble=true \
  "$@"
