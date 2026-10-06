#!/bin/bash
#
# Rung 2a as the ladder actually defines it -- deformable fusion, calibrated
# reference points, degradation curriculum -- on the published recipe.
#
# WHY THIS IS THE RUN
#
# This is the best model the project has produced, in either regime, and it has
# never been given a full training budget. Closed loop, thirty routes, the
# ladder's own campaign:
#
#     model                                intact   camera   lidar   censored
#     rung0     dense baseline              25.02    33.87   34.59   3/4/3
#     rung2a    deformable + curriculum     44.66    47.39   52.87   0/0/1
#     rung4     gated, weights 0.2          36.05    30.92   39.55   2/5/4
#     rung2a_dense  dense + curriculum      33.50      --     7.05   8/1
#
# Two things in that table decide this run. The deformable operator with
# calibrated references is worth about eleven points over the dense operator
# under the same curriculum -- and it is worth them where it matters most, in
# the censored count: zero routes killed by the harness timeout against eight.
# And the gate costs: rung 4 sits below rung 2a in all three conditions.
#
# The third thing is the reason for the recipe. Rung 2a scored 44.66 while
# trained at effective batch 8 for 10 epochs a stage. post31 -- the dense
# baseline at the published recipe, batch 64 for 31 epochs -- scored 43.20.
# The ladder's best model, on a third of the budget, already matches the
# corrected baseline. What it does with the full budget is the open question,
# and it is the one most likely to produce this thesis's best number.
#
# WHAT IT IS READ AGAINST
#
# results/closed_loop_rung2a_recipe.csv, the dense curriculum run at this same
# recipe. One design decision differs: the fusion operator and the reference
# points it starts from. That is two flags but one choice -- calibrated
# references exist only for the deformable operator, and the dense backbone
# ignores the knob.
#
# It is also the clean partner rung 3 has never had. Against this, a rung 3 at
# the same recipe differs by the head and the gate and nothing else, which is
# what makes the gate's result attributable for the first time.
#
# EXACTLY THE 44.66 MODEL, WITH A LONGER BUDGET
#
# Checked against outputs/rung2a_curriculum_only_post/config.yaml: every
# deformable knob it recorded -- num_points 4, learn_cross_reference true,
# reference_height_meter 0.8 -- is the code's default today, and only
# calibrated_reference differs from the default. So the two overrides below
# reproduce that model exactly, and nothing else is silently along for the ride.
#
# MEMORY
#
# rung2d is the anchor: deformable with calibrated references, batch 8, peak
# 6.2-6.3 GB, against the dense rung0's 6.1 GB at the same batch. The operator
# costs two or three per cent. The dense recipe at batch 32 x accum 2 peaks at
# 23.4 GB, so this should land near 24 on a 40 GB card with about 1.2 GB held
# by another user. Batch 64 was measured at 37.3 GB and rejected for that
# reason; this is not that.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	PRE=$HOME/LEAD/lead/outputs/rung2ad_lead_recipe
	POST=$HOME/LEAD/lead/outputs/rung2ad_lead_recipe_post
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_rung2ad_recipe.csv
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
		say "stage 1: 31-epoch pretrain, batch 32 x accum 2, deformable + curriculum"
		$PY src/lead/training/train.py \
			policy.transfuser.use_planning_decoder=false \
			policy.transfuser.backbone_target="$DEFORMABLE" \
			policy.transfuser.deformable_calibrated_reference=true \
			training.data.use_sensor_degradation=true \
			training.experiment.output_dir="$PRE" \
			training.experiment.resume_from_last_checkpoint=true \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung2ad_recipe_pretrain.log 2>&1
	fi

	PRE_CKPT=$(ls -t "$PRE"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$PRE_CKPT" ]; then
		say "FATAL: stage 1 produced no checkpoint"
		report_oom ~/rung2ad_recipe_pretrain.log
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
			training.data.use_sensor_degradation=true \
			training.experiment.initial_weights_file="$PRE_CKPT" \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.output_dir="$POST" \
			training.optimization.num_epochs=31 \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			>> ~/rung2ad_recipe_train.log 2>&1
	fi

	POST_CKPT=$(ls -t "$POST"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$POST_CKPT" ]; then
		say "FATAL: stage 2 produced no checkpoint; not evaluating"
		report_oom ~/rung2ad_recipe_train.log
		exit 1
	fi
	say "stage 2 done: $(basename "$POST_CKPT")"

	# --- score it on the same routes as post31 and rung 2a ----------------
	# --out keeps existing rows and skips them, so this survives a restart.
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models rung2ad_recipe=outputs/rung2ad_lead_recipe_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/rung2ad_recipe_eval.log 2>&1

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_rung2a_recipe.csv (dense, same recipe)"
	say "and results/closed_loop_post31.csv (the corrected baseline)."
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
