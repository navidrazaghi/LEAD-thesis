#!/bin/bash
#
# Rung 3 -- the full method -- on the published recipe instead of the ladder's.
#
# WHY THIS RUN EXISTS
#
# The gate is the thesis's own contribution and its result is negative. That
# negative was measured entirely in the ladder's regime: effective batch 8, 10
# epochs a stage. post31 showed what that regime does to a model. The intact
# score went from 18.77 to 43.20 and the stall rate from 73 to 33 per cent when
# the same baseline was trained at the published recipe -- and in that regime,
# destroying the LiDAR *raised* the baseline's score, because the noise broke a
# stall the model could not break itself. Every degradation column the gate was
# judged on is read through that. A null result from a sick instrument is not a
# null result.
#
# WHAT IT IS READ AGAINST
#
# results/closed_loop_rung2ad_recipe.csv -- rung 2a with the deformable
# operator at this same recipe, which run_rung2ad_recipe.sh produces and which
# must therefore run first. Against that, this differs by use_observability and
# use_observability_gate and nothing else: same operator, same reference
# points, same curriculum, same budget, same data, same routes, same
# controller. Two flags, one mechanism -- the head and the steering that reads
# it -- and the ladder's rung 2b is what separates those two in the old regime.
#
# That pairing is the whole reason rung 2a deformable is run first. Against the
# dense curriculum run it would have been a three-flag jump and a loss would
# have been unattributable.
#
# WHAT TO EXPECT, SO THE RESULT IS NOT A SURPRISE
#
# The ladder's own closed-loop campaign says this will probably lose. rung 4 --
# the gated model at diluted weights -- scored 36.05 / 30.92 / 39.55 where
# rung 2a scored 44.66 / 47.39 / 52.87, and rung 3 at full weight was worse
# than rung 4 in open loop. The value of running it is not the hope of a win.
# It is that a loss here is a loss in a regime where the model drives, which is
# a statement the thesis currently cannot make.
#
# One thing that can be said already, without the GPU: the head and the gate
# add 40,914 trainable parameters to a 67.7 M model, six hundredths of a per
# cent. Whatever the deficit is, it is not capacity. It is the auxiliary losses
# competing with the driving losses, which is what rung 4 was built to test.
#
# The gate objective is left at its default, "logit". The centred-log objective
# committed in 2666aa63 is for the next machine: turning it on here would
# change two things at once and make this row unreadable against old rung 3.
#
# WHAT IS ALREADY CHECKED
#
# The 450-log cache carries the observability targets -- `observability` and
# `observability_mask` are keys in every log's LMDB, written when rung 2b built
# them on 2026-09-20 -- so use_observability=true does not trigger a rebuild of
# 450 logs on a disk with 13 GB free.
#
# Both stages of this exact override list were built on CPU before this script
# was trusted: the config resolves, build_policy returns, gates_fusion is True,
# and the observability decoder and its token targets are present. The same
# check confirmed that the dense backbone refuses the gate rather than ignoring
# it, which is why rung 2a deformable and not the dense run is the partner
# above.
#
# MEMORY
#
# rung2d is the anchor: deformable with calibrated references at batch 8 peaks
# at 6.2-6.3 GB against the dense rung0's 6.1 GB. The head and gate add almost
# no parameters. The dense recipe at batch 32 x accum 2 peaks at 23.4 GB, so
# this should land near 24 on a 40 GB card with about 1.2 GB held by another
# user. Batch 64 was measured at 37.3 GB and rejected; this is not that.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	PRE=$HOME/LEAD/lead/outputs/rung3_lead_recipe
	POST=$HOME/LEAD/lead/outputs/rung3_lead_recipe_post
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_rung3_recipe.csv
	PARTNER=results/closed_loop_rung2ad_recipe.csv
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

	# Say so if the partner is missing, but do not refuse. The row is still
	# worth having against the dense run and post31; it is only harder to read.
	if [ ! -s "$PARTNER" ]; then
		say "NOTE: $PARTNER does not exist yet."
		say "  Without it this row is a three-flag comparison, not a two-flag one."
	fi

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
	#
	# The head and the gate both live in the encoder, so they must be present
	# from the pretrain. Adding them at post-train time would give them a third
	# of the budget every other part of the model had, and a null result would
	# not be readable.
	if [ -n "$(ls "$PRE"/model_*.pth 2>/dev/null)" ]; then
		say "stage 1 already has a checkpoint, skipping"
	else
		say "stage 1: 31-epoch pretrain, batch 32 x accum 2, head + gate + curriculum"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=false \
			policy.transfuser.backbone_target="$DEFORMABLE" \
			policy.transfuser.deformable_calibrated_reference=true \
			policy.transfuser.use_observability=true \
			policy.transfuser.use_observability_gate=true \
			training.data.use_sensor_degradation=true \
			training.experiment.output_dir="$PRE" \
			training.experiment.resume_from_last_checkpoint=true \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung3_recipe_pretrain.log 2>&1
	fi

	PRE_CKPT=$(ls -t "$PRE"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$PRE_CKPT" ]; then
		say "FATAL: stage 1 produced no checkpoint"
		report_oom ~/rung3_recipe_pretrain.log
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
			policy.transfuser.deformable_calibrated_reference=true \
			policy.transfuser.use_observability=true \
			policy.transfuser.use_observability_gate=true \
			training.data.use_sensor_degradation=true \
			training.experiment.initial_weights_file="$PRE_CKPT" \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.output_dir="$POST" \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung3_recipe_train.log 2>&1
	fi

	POST_CKPT=$(ls -t "$POST"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$POST_CKPT" ]; then
		say "FATAL: stage 2 produced no checkpoint; not evaluating"
		report_oom ~/rung3_recipe_train.log
		exit 1
	fi
	say "stage 2 done: $(basename "$POST_CKPT")"

	# --- score it on the same routes as every other run -------------------
	# --out keeps existing rows and skips them, so this survives a restart.
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models rung3_recipe=outputs/rung3_lead_recipe_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/rung3_recipe_eval.log 2>&1

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against $PARTNER, paired by route: two flags, one mechanism."
}

# Wait for our own processes to leave the GPU, then start.
#
# Waiting rather than refusing, because this is chained behind the rung 2a
# deformable run whose evaluation may still be tearing down a CARLA server.
# Only ours: the card is shared, and user omati has held about 1.2 GB with a
# YOLO process for over a day, so a guard that refuses on any compute app would
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
# comparable to the runs this is being read against.
report_oom() {
	local log="$1"
	if grep -qi "out of memory" "$log" 2>/dev/null; then
		say "  the failure was CUDA OUT OF MEMORY. Do not simply lower the batch:"
		say "  batch is part of the recipe being compared. Report it first."
	fi
	tail -20 "$log"
}

main "$@"
