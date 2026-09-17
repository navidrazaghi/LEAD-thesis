#!/bin/bash
#
# Seed replicate of dense + curriculum v2 + consistency 0.1, the whole pipeline.
#
# Same as run_diverse_dense_curriculum2.sh (pretrain) followed by
# run_diverse_dense_consistency.sh (post-train and scoring), with one change in
# both stages:
#
#   training.experiment.seed=1        (the seed-0 runs used the default 0)
#
# pl.seed_everything takes that seed, so weight init of the fresh heads, data
# order, dropout and the training-time damage draws all change. Evaluation is
# untouched: evaluation.inference.degrade_seed stays 0, so the damage the model
# is scored under is the same stream as in the seed-0 evaluation.
#
# Read against results/closed_loop_diverse_dense_consistency.csv (seed 0).
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung2a_diverse_curriculum2_seed1
	POST=$HOME/LEAD/lead/outputs/rung2a_diverse_consistency_seed1_post31
	CSV=results/closed_loop_diverse_dense_consistency_seed1.csv
	LOG=$HOME/diverse_dense_consistency_seed1.log
	SHARDS=${SHARDS:-3}
	SEED=1

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	if pgrep -f '[l]ead.training.train' > /dev/null || pgrep -f '[e]val_parallel.py' > /dev/null; then
		say "FATAL: a training or evaluation is already running; not sharing the GPU"
		exit 1
	fi

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
		training.experiment.seed=$SEED
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
		say "pretrain starting: 31 epochs, dense + curriculum v2, seed $SEED"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" \
			>> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"

	if [ ! -f "$POST/model_0030.pth" ]; then
		say "post-train starting: 31 epochs, curriculum v2 + consistency 0.1, seed $SEED"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.data.degradation_consistency_weight=0.1 \
			policy.transfuser.use_planning_decoder=true \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.initial_weights_file="$PRE/model_0030.pth" \
			training.experiment.output_dir="$POST" \
			>> "$LOG" 2>&1
		[ -f "$POST/model_0030.pth" ] || { say "FATAL: post-train ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "post-train done"

	say "scoring dense_consistency_seed1, 30 routes x 3 conditions, $SHARDS instance(s)"
	$PY ~/eval_parallel.py --models dense_consistency_seed1="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" --shards "$SHARDS" --base-port 8800 \
		|| say "WARNING: dense_consistency_seed1 scoring reported missing rows; see outputs/eval_shards"
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_dense_consistency.csv: only the seed differs."
}

main "$@"
