#!/bin/bash
#
# Rung 2c: rung 2a with the cross-modal hallucination head, and nothing else.
#
# Rung 2a is the best rung the ladder has, and the only one whose gain is not in
# dispute. It changes what the encoders learn. The gate, which changes only how
# the fusion routes between them, did not help. If that split is the real one,
# rebuilding a destroyed LiDAR grid from the camera should land on the working
# side of it -- and rung 2a is the honest thing to build it on, because a gain
# there is the head's own rather than a rescue of something already failing.
#
# Every setting below is copied from the config rung 2a recorded, so the two
# runs differ in one flag. That includes batch 8 and 10 epochs, which are the
# ladder's recipe rather than the published one: comparing against rung 2a
# matters more here than matching LEAD, and mixing the two would confound the
# head with the recipe fix.
#
# Both stages, not just the post-train. The head lives in the backbone, so a
# post-train-only run would give it a third of the budget every other part of
# rung 2a had, and a null result would not be readable.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	PRE=$HOME/LEAD/lead/outputs/rung2c_hallucination
	POST=$HOME/LEAD/lead/outputs/rung2c_hallucination_post
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	CSV=results/closed_loop_rung2c.csv
	BACKBONE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false
	export TIMM_USE_OLD_CACHE=1
	export WANDB_MODE=offline
	# 8 workers with pinned memory exceed the default 1024 descriptors; the
	# first epoch dies on "Too many open files" and the trainer hangs at step 0.
	ulimit -n 65536

	# The 28 held-out logs for the observability evaluation sit in the same tree
	# and py123d_log_names defaults to every log on disk. Naming the cached 450
	# restores the ladder's training set and keeps the held-out set held out.
	LOG_COUNT=$(ls "$CACHE" | wc -l)
	if [ "$LOG_COUNT" -ne 450 ]; then
		say "FATAL: expected 450 cached logs, found $LOG_COUNT"
		exit 1
	fi
	LOG_NAMES=$(ls "$CACHE" | sort | paste -sd, -)

	say "waiting for the post31 chain to finish"
	while pgrep -u "$USER" -f "training/train.py" >/dev/null ||
		pgrep -u "$USER" -f "run_evaluation.py" >/dev/null; do
		sleep 120
	done
	say "the machine is free"
	sleep 30

	common=(
		"policy.transfuser.backbone_target=$BACKBONE"
		"policy.transfuser.deformable_calibrated_reference=true"
		"policy.transfuser.use_cross_modal_hallucination=true"
		"training.data.use_sensor_degradation=true"
		"training.optimization.num_epochs=10"
		"training.optimization.batch_size=8"
		"training.lightning.accumulate_grad_batches=1"
		"training.data.read_from_cache_store=true"
		"training.data.py123d_log_names=[$LOG_NAMES]"
	)

	say "pretrain starting"
	$PY src/lead/training/train.py \
		"${common[@]}" \
		policy.transfuser.use_planning_decoder=false \
		training.experiment.resume_from_last_checkpoint=false \
		"training.experiment.output_dir=$PRE" \
		>> ~/rung2c_train.log 2>&1

	PRE_CKPT=$(ls -t "$PRE"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$PRE_CKPT" ]; then
		say "FATAL: pretrain produced no checkpoint"
		tail -20 ~/rung2c_train.log
		exit 1
	fi
	say "pretrain done, $(basename "$PRE_CKPT")"

	say "post-train starting"
	$PY src/lead/training/train.py \
		"${common[@]}" \
		policy.transfuser.use_planning_decoder=true \
		training.experiment.resume_from_last_checkpoint=false \
		"training.experiment.initial_weights_file=$PRE_CKPT" \
		"training.experiment.output_dir=$POST" \
		>> ~/rung2c_train.log 2>&1

	POST_CKPT=$(ls -t "$POST"/model_*.pth 2>/dev/null | head -1)
	if [ -z "$POST_CKPT" ]; then
		say "FATAL: post-train produced no checkpoint"
		tail -20 ~/rung2c_train.log
		exit 1
	fi
	say "post-train done, $(basename "$POST_CKPT")"

	# Two evaluations, because the head and its use are separate questions.
	# Without the substitution this measures whether carrying the auxiliary task
	# costs anything; with it, whether rebuilding the grid buys anything.
	say "scoring without the substitution"
	$PY scripts/common/run_evaluation.py \
		--models rung2c=outputs/rung2c_hallucination_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> ~/rung2c_eval.log 2>&1

	say "scoring with the substitution, LiDAR column only"
	$PY scripts/common/run_evaluation.py \
		--models rung2c_halluc=outputs/rung2c_hallucination_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions lidar:1.0 \
		--out results/closed_loop_rung2c_substituted.csv \
		--config evaluation.inference.hallucinate_missing_lidar=true \
		>> ~/rung2c_eval.log 2>&1

	say "done"
	say "read against rung2a in results/closed_loop.csv: same recipe, same data,"
	say "same routes, one flag."
}

main "$@"
