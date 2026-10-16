#!/bin/bash
#
# Degradation-consistency self-distillation on the dense baseline, curriculum v2.
#
# One factor against rung2a_diverse_curriculum2_post31. The consistency term needs
# the planning outputs, which exist only in post-training, so this run takes that
# run's own pretrain (rung2a_diverse_curriculum2/model_0030.pth) and post-trains
# from it with every setting copied -- same 585 logs, curriculum v2 flags, 31
# epochs, batch 32 x accumulation 2 -- plus
#
#   training.data.degradation_consistency_weight = 0.1
#
# 0.1 is not tuned: the term measured 0.16 on a real damaged batch against a total
# training objective near 0.13 at the end of post-training, so weight 1 would
# swamp imitation. The logged losses/degradation_consistency is read early in the
# run to see whether that holds.
#
# Waits for the other queued chains, so the GPU is never shared between trainings,
# and refuses to start if the dense curriculum-v2 run did not finish.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung2a_diverse_curriculum2
	POST=$HOME/LEAD/lead/outputs/rung2a_diverse_consistency_post31
	CSV=results/closed_loop_diverse_dense_consistency.csv
	LOG=$HOME/diverse_dense_consistency.log
	SHARDS=${SHARDS:-3}

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	while pgrep -f '[r]un_parallel_pipeline.sh' > /dev/null \
		|| pgrep -f '[r]un_diverse_curriculum2_parallel.sh' > /dev/null \
		|| pgrep -f '[r]un_diverse_dense_curriculum2.sh' > /dev/null; do
		sleep 300
	done
	if [ ! -f "$PRE/model_0030.pth" ] || [ ! -f results/closed_loop_diverse_dense_curriculum2.csv ]; then
		say "FATAL: the dense curriculum-v2 run did not finish; see ~/diverse_dense_curriculum2_chain.log"
		exit 1
	fi
	say "dense curriculum-v2 run finished; starting consistency post-train from its pretrain"

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

	if [ ! -f "$POST/model_0030.pth" ]; then
		say "post-train starting: 31 epochs, curriculum v2 + consistency 0.1"
		$PY -m lead.training.train \
			training.data.use_sensor_degradation=true \
			training.data.sensor_degradation_probability=0.30 \
			training.data.sensor_degradation_independent_modalities=true \
			training.data.sensor_degradation_full_failure_probability=0.25 \
			training.data.sensor_degradation_misalignment_probability=0.10 \
			training.data.degradation_consistency_weight=0.1 \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.optimization.num_epochs=31 \
			policy.transfuser.use_planning_decoder=true \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.initial_weights_file="$PRE/model_0030.pth" \
			training.experiment.output_dir="$POST" \
			>> "$LOG" 2>&1
		[ -f "$POST/model_0030.pth" ] || { say "FATAL: post-train ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "post-train done"

	say "scoring dense_consistency, 30 routes x 3 conditions, $SHARDS instance(s)"
	$PY ~/eval_parallel.py --models dense_consistency="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" --shards "$SHARDS" --base-port 8800 \
		|| say "WARNING: dense_consistency scoring reported missing rows; see outputs/eval_shards"
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_dense_curriculum2.csv: same pretrain,"
	say "same curriculum, same recipe -- only the consistency term differs."
}

main "$@"
