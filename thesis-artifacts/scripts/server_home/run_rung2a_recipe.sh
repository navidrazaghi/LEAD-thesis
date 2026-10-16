#!/bin/bash
#
# Rung 2a's degradation curriculum, on the published recipe instead of the
# ladder's.
#
# The ladder is two steps away from the recipe LEAD publishes, not one. Every
# rung trained at an effective batch of 8 -- batch 8 with no accumulation --
# for 10 epochs a stage, against the published 64 for 31. post31 closed the
# second gap for the baseline alone and the result was not cosmetic: the
# intact score went from 18.77 to 43.20, the stall rate from 73 to 33 per
# cent, and sensor damage started costing score instead of adding it. In the
# ladder's regime, destroying the LiDAR *raised* the baseline's score, because
# the noise broke a stall the model could not break itself. That is a sick
# measurement, and every degradation column in the ladder is read through it.
#
# So this reruns the one rung whose gain the thesis actually claims, at the
# recipe that produces a model which drives. Against post31 it is a clean
# paired comparison: same pretrain budget, same post-train budget, same batch,
# same data, same routes, same controller, and exactly one flag different --
# use_sensor_degradation. post31 is the corrected rung 0; this is the
# corrected rung 2a.
#
# Both stages carry the curriculum, because rung 2a did. Reusing the
# rung0_lead_recipe pretrain would give the curriculum a third of the budget
# the rest of the model had and make a null result unreadable.
#
# What this cannot do in the time left is rerun rung 4. The gate's result
# stays in the old regime, and the thesis has to say so.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	PRE=$HOME/LEAD/lead/outputs/rung2a_lead_recipe
	POST=$HOME/LEAD/lead/outputs/rung2a_lead_recipe_post
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_rung2a_recipe.csv

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
	# Without this the trainer stops on the first line asking for an API key.
	export WANDB_MODE=offline

	# 8 workers with pinned memory and a prefetch of 2 pass tensors as file
	# descriptors and exceed the default soft limit of 1024: the first epoch
	# dies on "Too many open files" and the trainer then hangs at step 0 with
	# the GPU idle, which reads like a deadlock rather than a limit.
	ulimit -n 65536

	# Train on exactly the logs the cache was built for, and no others.
	#
	# py123d_log_names defaults to empty, which means every log on disk. The 28
	# held-out logs for the observability evaluation were downloaded into the
	# same tree on 2026-10-01, so the default now silently trains on them and
	# invalidates that result with no error to show for it. The rung 2a this is
	# being compared against predates the download and was unaffected; naming
	# the 450 is what keeps this run on the same training set it had.
	LOG_COUNT=$(ls "$CACHE" | wc -l)
	if [ "$LOG_COUNT" -ne 450 ]; then
		say "FATAL: expected 450 cached logs, found $LOG_COUNT"
		exit 1
	fi
	LOG_NAMES=$(ls "$CACHE" | sort | paste -sd, -)
	say "training set pinned to the $LOG_COUNT cached logs"

	# --- stage 1: pretrain, perception heads only -------------------------
	if [ -n "$(ls "$PRE"/model_*.pth 2>/dev/null)" ]; then
		say "stage 1 already has a checkpoint, skipping"
	else
		say "stage 1: 31-epoch pretrain, batch 32 x accum 2, curriculum on"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=false \
			training.data.use_sensor_degradation=true \
			training.experiment.output_dir="$PRE" \
			training.experiment.resume_from_last_checkpoint=true \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung2a_recipe_pretrain.log 2>&1
	fi

	PRE_CKPT=$(ls -t "$PRE"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$PRE_CKPT" ]; then
		say "FATAL: stage 1 produced no checkpoint"
		tail -20 ~/rung2a_recipe_pretrain.log
		exit 1
	fi
	say "stage 1 done: $(basename "$PRE_CKPT")"

	# --- stage 2: post-train, adds the planning decoder -------------------
	if [ -n "$(ls "$POST"/model_*.pth 2>/dev/null)" ]; then
		say "stage 2 already has a checkpoint, skipping"
	else
		say "stage 2: 31-epoch post-train from stage 1, curriculum on"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=true \
			training.data.use_sensor_degradation=true \
			training.experiment.initial_weights_file="$PRE_CKPT" \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.output_dir="$POST" \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung2a_recipe_train.log 2>&1
	fi

	POST_CKPT=$(ls -t "$POST"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$POST_CKPT" ]; then
		say "FATAL: stage 2 produced no checkpoint; not evaluating"
		tail -20 ~/rung2a_recipe_train.log
		exit 1
	fi
	say "stage 2 done: $(basename "$POST_CKPT")"

	# --- score it against post31 on the same routes -----------------------
	# --out keeps existing rows and skips them, so this survives a restart.
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models rung2a_recipe=outputs/rung2a_lead_recipe_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/rung2a_recipe_eval.log 2>&1

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_post31.csv, paired by route:"
	say "  same recipe, same data, same routes, one flag."
}

main "$@"
