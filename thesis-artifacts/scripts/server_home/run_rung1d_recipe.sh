#!/bin/bash
#
# Rung 2a deformable with the calibrated reference points turned off.
#
# WHY THIS RUN EXISTS
#
# rung1d_recipe scored 59.74 under full LiDAR destruction against the dense
# curriculum run's 31.07 -- the largest single gain this project has measured.
# But that gain belongs to a package: the deformable operator and the
# calibrated reference points arrived together, and nothing separates them at
# the published recipe.
#
# The ladder's open-loop table says the separation matters, and says it in the
# uncomfortable direction. Waypoint L2, lower better:
#
#     rung 1  deformable, geometry-free   0.492 clean   0.675 cam   1.005 lidar
#     rung 2  deformable, calibrated      0.555 clean   1.093 cam   1.643 lidar
#
# Calibrated was worse in all three columns. Only with the curriculum on top
# (rung 2a, 0.442) did it become the best rung in the ladder. So the evidence
# points at an interaction rather than at the prior being good on its own, and
# a thesis that claims the calibrated reference points as a contribution has to
# be able to say which it is.
#
# This is the control that answers it. One flag from rung1d_recipe:
# deformable_calibrated_reference, which is not passed here because false is
# the code's own default. Everything else -- operator, curriculum, budget,
# data, routes, controller -- is identical.
#
# WHAT MAKES THE COMPARISON CLEAN
#
# Checked on CPU before this ran: the calibrated flag changes no parameter
# count at all. Both configurations build to 67,673,140 trainable parameters.
# The reference points decide where a query starts reading, not how much
# capacity it has, so any difference between this run and rung1d_recipe is
# the geometric prior and nothing else.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	PRE=$HOME/LEAD/lead/outputs/rung1d_lead_recipe
	POST=$HOME/LEAD/lead/outputs/rung1d_lead_recipe_post
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_rung1d_recipe.csv
	DEFORMABLE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

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

	wait_for_our_gpu_to_clear || exit 1
	check_disk || exit 1

	# Train on exactly the logs the cache was built for, and no others.
	#
	# py123d_log_names defaults to empty, which means every log on disk. The 28
	# held-out logs for the observability evaluation were downloaded into the
	# same tree on 2026-10-01, so the default now silently trains on them and
	# invalidates that result with no error to show for it.
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
		say "stage 1: 31-epoch pretrain, deformable geometry-free + curriculum"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=false \
			policy.transfuser.backbone_target="$DEFORMABLE" \
			training.data.use_sensor_degradation=true \
			training.experiment.output_dir="$PRE" \
			training.experiment.resume_from_last_checkpoint=true \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung1d_recipe_pretrain.log 2>&1
	fi

	PRE_CKPT=$(ls -t "$PRE"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$PRE_CKPT" ]; then
		say "FATAL: stage 1 produced no checkpoint"
		report_oom ~/rung1d_recipe_pretrain.log
		exit 1
	fi
	say "stage 1 done: $(basename "$PRE_CKPT")"

	# --- stage 2: post-train, adds the planning decoder -------------------
	if [ -n "$(ls "$POST"/model_*.pth 2>/dev/null)" ]; then
		say "stage 2 already has a checkpoint, skipping"
	else
		say "stage 2: 31-epoch post-train from stage 1, same flags"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=true \
			policy.transfuser.backbone_target="$DEFORMABLE" \
			training.data.use_sensor_degradation=true \
			training.experiment.initial_weights_file="$PRE_CKPT" \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.output_dir="$POST" \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung1d_recipe_train.log 2>&1
	fi

	POST_CKPT=$(ls -t "$POST"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$POST_CKPT" ]; then
		say "FATAL: stage 2 produced no checkpoint; not evaluating"
		report_oom ~/rung1d_recipe_train.log
		exit 1
	fi
	say "stage 2 done: $(basename "$POST_CKPT")"

	# --- score it on the same routes as post31 and rung 2a ----------------
	# --out keeps existing rows and skips them, so this survives a restart.
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models rung1d_recipe=outputs/rung1d_lead_recipe_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/rung1d_recipe_eval.log 2>&1

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_rung2ad_recipe.csv, paired by route:"
	say "  one flag apart -- the calibrated reference points, and nothing else."
}

# Wait for our own processes to leave the GPU, then start.
#
# Waiting rather than refusing, because this is meant to be chained behind
# another run whose evaluation may still be tearing down a CARLA server. Only
# ours: the card is shared, and user omati has held about 1.2 GB with a YOLO
# process for over a day, so a guard that refuses on any compute app would
# refuse forever.
wait_for_our_gpu_to_clear() {
	local waited=0
	local limit=3600
	while true; do
		local busy=""
		for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
			if [ "$(ps -o user= -p "$gpu_pid" 2>/dev/null | tr -d ' ')" = "$(id -un)" ]; then
				busy="$gpu_pid"
				break
			fi
		done
		[ -z "$busy" ] && break
		if [ "$waited" -ge "$limit" ]; then
			say "FATAL: pid $busy of ours has held the GPU for ${limit}s; not starting"
			ps -o pid=,etime=,cmd= -p "$busy" | cut -c1-120
			return 1
		fi
		[ "$waited" = 0 ] && say "waiting for our pid $busy to leave the GPU"
		sleep 60
		waited=$(( waited + 60 ))
	done
	say "GPU is ours"
	return 0
}

# Two stages of checkpoints are about 2.3 GB, and the evaluation writes its own
# logs beside them. Stopping here beats dying at epoch 20.
check_disk() {
	local free_gb
	free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
	if [ "$free_gb" -lt 6 ]; then
		say "FATAL: only ${free_gb}G free; need headroom for 2.3G of checkpoints"
		return 1
	fi
	say "disk: ${free_gb}G free"
	return 0
}

# Say plainly when a stage died for memory, because the fix differs from every
# other failure: the recipe would have to change, and a changed recipe is not
# comparable to post31 or to the dense curriculum run.
report_oom() {
	local log="$1"
	if grep -qi "out of memory" "$log" 2>/dev/null; then
		say "  the failure was CUDA OUT OF MEMORY. Do not simply lower the batch:"
		say "  batch is part of the recipe being compared. Report it first."
	fi
	tail -20 "$log"
}

main "$@"
