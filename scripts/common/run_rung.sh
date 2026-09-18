#!/bin/bash
# One ablation rung. Usage: run_rung.sh <output-name> [config overrides...]
#
# ulimit is not optional: the default 1024 starves the LMDB cache readers and
# training crawls at 1/50th speed without ever failing outright.
#
# resume_from_last_checkpoint also decides whether load_state_dict is strict
# (train.py:97). So it must be false on the first launch of a posttrain, whose
# initial_weights_file comes from a pretrain that has no planning decoder, and
# true only when resuming a run into its own architecture. Overrides land last,
# so passing it explicitly wins over the default here.
set -u
NAME=$1; shift
ulimit -n 65536
cd "$HOME/LEAD/lead"
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue WANDB_MODE=offline \
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
  LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}" \
  "$HOME/miniconda3/envs/lead/bin/python" -m lead.training.train \
  training.data.read_from_cache_store=true \
  training.optimization.batch_size=8 \
  training.optimization.num_epochs=10 \
  training.experiment.resume_from_last_checkpoint=true \
  training.experiment.output_dir="$HOME/LEAD/lead/outputs/$NAME" "$@"
