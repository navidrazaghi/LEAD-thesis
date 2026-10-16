#!/bin/bash
#
# The deformable operator on the diverse subset: one factor against rung0_diverse.
#
# Everything is copied from run_diverse_baseline.sh -- same 585 logs, same
# cache, same 31 + 31 epochs, same batch 32 x accumulation 2, same 30 scored
# routes -- and only the fusion operator changes:
#
#   backbone_target                  = DeformableFusionBackbone
#   deformable_calibrated_reference  = true   (rig geometry seeds the cross-modal
#                                              reference points)
#   deformable_learn_cross_reference = true
#   deformable_num_points            = 4
#
# The gate, the residual gain and the observability mask stay off, and so does
# the degradation curriculum. That last one is deliberate and differs from
# rung2ad, which carried the curriculum: with it on, this run would differ from
# rung0_diverse in two factors at once and the operator's own contribution would
# not be readable. The curriculum variant is a separate run if it is wanted.
#
# Body in a function called on the last line, so editing this file while it
# runs cannot feed bash shifted bytes.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung2ad_diverse
	POST=$HOME/LEAD/lead/outputs/rung2ad_diverse_post31
	CSV=results/closed_loop_diverse_deformable.csv
	LOG=$HOME/diverse_deformable.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	# --- the training set, pinned exactly as the baseline pinned it ---------
	NAMES=$(awk -F/ '{print $2}' "$SEL" | sort)
	N=$(echo "$NAMES" | wc -l)
	[ "$N" -eq 585 ] || { say "FATAL: $N logs in the selection, expected 585"; exit 1; }
	LOG_NAMES=$(echo "$NAMES" | paste -sd, -)
	[ -f "$HOME/LEAD/lead/outputs/rung0_diverse_post31/model_0030.pth" ] || {
		say "FATAL: the diverse baseline is missing; there is nothing to compare against"
		exit 1
	}

	DEFORMABLE=(
		policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone
		policy.transfuser.deformable_calibrated_reference=true
		policy.transfuser.deformable_learn_cross_reference=true
		policy.transfuser.deformable_num_points=4
	)
	COMMON=(
		"${DEFORMABLE[@]}"
		training.data.read_from_cache_store=true
		"training.data.py123d_log_names=[$LOG_NAMES]"
		training.optimization.batch_size=32
		training.lightning.accumulate_grad_batches=2
		training.optimization.num_epochs=31
	)

	# --- pretrain -----------------------------------------------------------
	if [ ! -f "$PRE/model_0030.pth" ]; then
		say "pretrain starting: 31 epochs, deformable + calibrated reference"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" \
			>> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"

	# --- post-train ---------------------------------------------------------
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

	# --- evaluation ---------------------------------------------------------
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models deformable_diverse="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> "$LOG" 2>&1
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_baseline.csv: same data, same"
	say "recipe, same routes -- the fusion operator is the only difference."
}

main "$@"
