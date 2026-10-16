#!/bin/bash
#
# The new baseline plus curriculum v2: one factor against rung0_diverse.
#
# Completes the 2 x 2 on the diverse subset:
#
#                      no curriculum         curriculum v2
#   dense (baseline)   rung0_diverse         THIS RUN
#   deformable         rung2ad_diverse       rung2ad_diverse_curriculum2
#
# Everything is rung0_diverse's -- the default TransfuserBackbone (dense fusion
# attention), the same 585 logs and cache, 31 + 31 epochs, batch 32 x
# accumulation 2, the same 30 routes x 3 conditions -- and the only change is the
# degradation curriculum, set exactly as in rung2ad_diverse_curriculum2 so the two
# curriculum rows differ in the operator alone.
#
# Waits for run_parallel_pipeline.sh (which trains and scores the curriculum-v2
# deformable run) to finish, and refuses to start if that run did not produce
# its scores, so the GPU is never shared between two trainings.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung2a_diverse_curriculum2
	POST=$HOME/LEAD/lead/outputs/rung2a_diverse_curriculum2_post31
	CSV=results/closed_loop_diverse_dense_curriculum2.csv
	LOG=$HOME/diverse_dense_curriculum2.log
	SHARDS=${SHARDS:-3}

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	while pgrep -f '[r]un_parallel_pipeline.sh' > /dev/null \
		|| pgrep -f '[r]un_diverse_curriculum2_parallel.sh' > /dev/null; do
		sleep 300
	done
	if [ ! -f results/closed_loop_diverse_curriculum2.csv ] \
		|| [ ! -f outputs/rung2ad_diverse_curriculum2_post31/model_0030.pth ]; then
		say "FATAL: the curriculum-v2 deformable run did not finish; see ~/parallel_pipeline.log"
		exit 1
	fi
	say "curriculum-v2 deformable run finished; starting dense + curriculum v2"

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
		say "pretrain starting: 31 epochs, dense + curriculum v2"
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

	say "scoring dense_curriculum2, 30 routes x 3 conditions, $SHARDS instance(s)"
	$PY ~/eval_parallel.py --models dense_curriculum2="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" --shards "$SHARDS" --base-port 7600 \
		|| say "WARNING: dense_curriculum2 scoring reported missing rows; see outputs/eval_shards"
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_baseline.csv: only the curriculum differs."
}

main "$@"
