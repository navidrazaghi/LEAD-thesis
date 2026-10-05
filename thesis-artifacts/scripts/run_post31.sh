#!/bin/bash
#
# Post-train for 31 epochs instead of 10, then score it.
#
# The published LEAD recipe runs both stages at 31 epochs. Ours matched the
# pretrain and left the post-train at the ladder-wide 10, so the planning
# decoder -- the stage that actually drives -- got a third of the budget the
# perception encoder got. The retrained baseline stalls on 78% of runs against
# the reference checkpoint's 4%, and this is the cheapest hypothesis that
# explains it.
#
# Nothing else changes: every other setting is copied from the config the
# 10-epoch post-train recorded, so the two runs differ in one number.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	OUT=$HOME/LEAD/lead/outputs/rung0_lead_recipe_post31
	WEIGHTS=$HOME/LEAD/lead/outputs/rung0_lead_recipe/model_0030.pth
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_post31.csv

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	# From scripts/common/pretrain.sh: one data loader worker per core already,
	# so the numeric libraries must not start threads of their own.
	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	# Beartype and Dynamo cannot run together.
	export LEAD_RUNTIME_TYPE_CHECKING=false
	# Hugging Face is unreachable from here; the weights are in ~/.cache/torch.
	export TIMM_USE_OLD_CACHE=1
	# Every run in this project logs to W&B offline. Without this the trainer
	# stops on the first line asking for an API key.
	export WANDB_MODE=offline

	# The dataloader passes tensors between workers as file descriptors, and the
	# default soft limit of 1024 is not enough for 8 workers with pinned memory
	# and a prefetch of 2: the first epoch dies on "Too many open files" and the
	# trainer then hangs at step 0 with the GPU idle. The hard limit here is
	# 1048576, so raising the soft one needs no privilege.
	ulimit -n 65536

	# Train on exactly the logs the cache was built for, and no others.
	#
	# py123d_log_names defaults to empty, which means "every log on disk". That
	# was 450 when the 10-epoch post-train ran. On 2026-10-01 the 28 held-out
	# logs for the observability evaluation were downloaded into the same tree,
	# so the disk now carries 478 and the loader asks for cache entries that
	# were never built -- which is how this surfaced, as an lmdb "No such file".
	#
	# Naming the 450 does two things: it restores the exact training set the run
	# being compared against used, and it keeps the held-out 28 held out.
	# Training on them would have quietly invalidated the observability result,
	# with no error to show for it.
	LOG_COUNT=$(ls "$CACHE" | wc -l)
	if [ "$LOG_COUNT" -ne 450 ]; then
		say "FATAL: expected 450 cached logs, found $LOG_COUNT"
		exit 1
	fi
	LOG_NAMES=$(ls "$CACHE" | sort | paste -sd, -)
	say "training set pinned to the $LOG_COUNT cached logs"

	if [ ! -f "$WEIGHTS" ]; then
		say "FATAL: pretrain weights missing at $WEIGHTS"
		exit 1
	fi

	say "post-train starting: 31 epochs from the 31-epoch pretrain"
	say "  the 10-epoch run is outputs/rung0_lead_recipe_post, kept intact"

	$PY src/lead/training/train.py \
		policy.transfuser.use_planning_decoder=true \
		training.experiment.initial_weights_file="$WEIGHTS" \
		training.experiment.resume_from_last_checkpoint=false \
		training.experiment.output_dir="$OUT" \
		training.optimization.num_epochs=31 \
		training.optimization.batch_size=32 \
		training.lightning.accumulate_grad_batches=2 \
		training.data.read_from_cache_store=true \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		>> ~/post31_train.log 2>&1

	CKPT=$(ls -t "$OUT"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$CKPT" ]; then
		say "FATAL: training produced no checkpoint; not evaluating"
		tail -20 ~/post31_train.log
		exit 1
	fi
	say "training done, checkpoint $(basename "$CKPT")"

	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models post31=outputs/rung0_lead_recipe_post31 \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/post31_eval.log 2>&1

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_new_baseline_idle.csv -- same pretrain,"
	say "same data, same machine, only the post-train budget differs."
}

main "$@"
