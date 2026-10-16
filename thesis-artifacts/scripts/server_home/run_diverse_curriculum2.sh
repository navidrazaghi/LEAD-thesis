#!/bin/bash
#
# The widened degradation curriculum, on top of the deformable operator.
#
# Waits for run_diverse_deformable.sh to finish training and scoring, then
# trains the same model on the same 585 logs with the curriculum turned on in
# its v2 form:
#
#   independent_modalities   = true  -> 49/21/21/9 clean/camera/lidar/both
#   probability              = 0.30    (arXiv 2603.05623's split)
#   full_failure_probability = 0.25  -> the fully-failed case, which the
#                                       closed-loop protocol scores, enters
#                                       training with positive probability
#                                       (Grace-BEV, CMT masked-modal training)
#   misalignment_probability = 0.10  -> camera/LiDAR geometry perturbed, which
#                                       MultiCorrupt finds among the most
#                                       damaging and this stack never saw
#
# One factor against rung2ad_diverse: same data, same operator, same recipe,
# and only the curriculum differs. It is NOT comparable to rung2a or rung2ad of
# the thesis ladder, which used the exclusive single-modality curriculum and a
# different training subset.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	OPERATOR=$HOME/LEAD/lead/outputs/rung2ad_diverse_post31
	PRE=$HOME/LEAD/lead/outputs/rung2ad_diverse_curriculum2
	POST=$HOME/LEAD/lead/outputs/rung2ad_diverse_curriculum2_post31
	CSV=results/closed_loop_diverse_curriculum2.csv
	LOG=$HOME/diverse_curriculum2.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	while pgrep -f '[r]un_diverse_deformable.sh' > /dev/null; do sleep 300; done
	if [ ! -f "$OPERATOR/model_0030.pth" ] || [ ! -f results/closed_loop_diverse_deformable.csv ]; then
		say "FATAL: the deformable chain did not finish; see ~/diverse_deformable_chain.log"
		exit 1
	fi
	say "deformable chain finished; starting the widened curriculum"

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	NAMES=$(awk -F/ '{print $2}' "$SEL" | sort)
	N=$(echo "$NAMES" | wc -l)
	[ "$N" -eq 585 ] || { say "FATAL: $N logs in the selection, expected 585"; exit 1; }
	LOG_NAMES=$(echo "$NAMES" | paste -sd, -)

	COMMON=(
		policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone
		policy.transfuser.deformable_calibrated_reference=true
		policy.transfuser.deformable_learn_cross_reference=true
		policy.transfuser.deformable_num_points=4
		training.data.use_sensor_degradation=true
		training.data.sensor_degradation_probability=0.30
		training.data.sensor_degradation_independent_modalities=true
		training.data.sensor_degradation_full_failure_probability=0.25
		training.data.sensor_degradation_misalignment_probability=0.10
		training.data.read_from_cache_store=true
		"training.data.py123d_log_names=[$LOG_NAMES]"
		training.optimization.batch_size=32
		training.lightning.accumulate_grad_batches=2
		training.optimization.num_epochs=31
	)

	if [ ! -f "$PRE/model_0030.pth" ]; then
		say "pretrain starting: 31 epochs, deformable + curriculum v2"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" \
			>> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"

	if [ ! -f "$POST/model_0030.pth" ]; then
		say "post-train starting: 31 epochs from the pretrain"
		$PY -m lead.training.train "${COMMON[@]}" \
			policy.transfuser.use_planning_decoder=true \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.initial_weights_file="$PRE/model_0030.pth" \
			training.experiment.output_dir="$POST" \
			>> "$LOG" 2>&1
		[ -f "$POST/model_0030.pth" ] || { say "FATAL: post-train ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "post-train done"

	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models curriculum2="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> "$LOG" 2>&1
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_deformable.csv: same data, same"
	say "operator, same recipe -- only the degradation curriculum differs."
}

main "$@"
