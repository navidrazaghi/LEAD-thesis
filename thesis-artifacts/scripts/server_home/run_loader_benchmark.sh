#!/bin/bash
#
# Is training bound by the data loader or by the GPU?
#
# THE QUESTION
#
# The last runs held 57.7 samples/s. Two ResNet-34 branches at 384x1152 and
# 320x384 plus the fusion transformers come to roughly 150 GFLOP per sample
# with the backward pass, so that rate is about 8.7 TFLOP/s against an A100
# that does 150-250 in bf16. Four percent of the card. nvidia-smi reads 100%,
# but that counts time with a kernel resident, not time doing useful work, and
# a starved GPU reads the same as a busy one.
#
# The suspect is num_assigned_cpu_cores, which returns SLURM_CPUS_PER_TASK when
# SLURM_JOB_ID is also set and otherwise its default of 8. There is no SLURM on
# this machine, so every run so far used 8 loader workers on 32 cores, and each
# sample costs three JPEG decodes, a LiDAR sweep accumulation, a BEV raster and
# the CenterNet and semantic label builds -- all on the CPU, all single-threaded
# because the dataset sets cv2.setNumThreads(0).
#
# WHY THIS DESIGN
#
# Three arms of the real trainer, not a reimplementation of the input pipeline.
# A hand-written loader loop would measure a pipeline nobody trains with.
#
#   A  8 workers               the configuration every result so far was made on
#   B  24 workers              the same run with the cores it could have had
#   C  24 workers, out of order  workers deliver as they finish
#
# If B beats A the loader was the bottleneck and the answer is a environment
# variable. If B matches A the GPU was saturated after all, the 4% figure is
# wrong somewhere, and a bigger dataset costs proportionally more time.
#
# 24 and not 32: CARLA and the shell need cores too, and a loader that starves
# the rest of the box is not a configuration anyone would ship.
#
# WHAT IT DOES NOT TOUCH
#
# Its checkpoints go to a scratch directory, it writes no CSV, and it reads the
# same 450-log cache every run has read. Deleting its output directory undoes
# it completely.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	CACHE=$HOME/LEAD/lead/data/lead/123D/transfuser_training_cache/normal_view
	OUT=$HOME/LEAD/lead/outputs/loader_benchmark
	LOG=$HOME/loader_benchmark.log
	DEFORMABLE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

	# Steps per arm. torch.compile spends the first minute or so tracing, and
	# the trainer prints a running average, so the tail of a 200-step run is a
	# settled number while a 60-step run is mostly warm-up.
	STEPS=200

	say() { echo "[$(date '+%m-%d %H:%M:%S')] bench: $*" | tee -a "$LOG"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false
	export TIMM_USE_OLD_CACHE=1
	export WANDB_MODE=offline
	# More workers means more shared-memory file descriptors, and the default
	# soft limit of 1024 is already too low at 8.
	ulimit -n 65536

	# Wait on the driver processes, never on the GPU: CARLA frees the card
	# between routes, so a GPU-idle test would fire in the middle of the
	# campaign. Two names because run_after_weather execs into
	# run_final_chain3, so one process changes its name partway through.
	QUEUE="run_after_weather[.]sh|run_final_chain3[.]sh|run_weather_recipe[.]sh"
	WORK="training/train[.]py|scripts/common/run_evaluation[.]py"

	say "waiting for the queue to drain"
	while pgrep -u "$USER" -f "$QUEUE" > /dev/null; do
		sleep 300
	done
	say "queue drivers have exited"
	while pgrep -u "$USER" -f "$WORK" > /dev/null; do
		sleep 120
	done
	say "no training or evaluation is running"
	sleep 30

	LOG_COUNT=$(ls "$CACHE" | wc -l)
	if [ "$LOG_COUNT" -ne 450 ]; then
		say "FATAL: expected 450 cached logs, found $LOG_COUNT"
		exit 1
	fi
	LOG_NAMES=$(ls "$CACHE" | sort | paste -sd, -)
	say "training set pinned to the $LOG_COUNT cached logs"
	say "$(df -h / | awk 'NR==2{print $4}') free on /"

	# The rung 2a-d configuration: the model a larger-dataset rerun would
	# actually use, and the one whose deformable operator is pure PyTorch and
	# so the most likely to be slow if the GPU is the bottleneck after all.
	common=(
		"policy.transfuser.use_planning_decoder=true"
		"policy.transfuser.backbone_target=$DEFORMABLE"
		"policy.transfuser.deformable_calibrated_reference=true"
		"training.data.use_sensor_degradation=true"
		"training.experiment.resume_from_last_checkpoint=false"
		"training.optimization.num_epochs=1"
		"training.optimization.batch_size=32"
		"training.lightning.accumulate_grad_batches=2"
		"training.lightning.max_steps=$STEPS"
		"training.data.read_from_cache_store=true"
		"training.data.py123d_log_names=[$LOG_NAMES]"
	)

	arm() {
		local name="$1" cores="$2" in_order="$3"
		local armlog="$HOME/bench_${name}.log"
		say "arm $name: $cores loader workers, in_order=$in_order, $STEPS steps"
		rm -rf "$OUT/$name"

		# num_assigned_cpu_cores reads SLURM_CPUS_PER_TASK only when
		# SLURM_JOB_ID is set too, so both are needed to move it off its
		# default of 8. No code change, and nothing persists past this shell.
		if [ "$cores" = "default" ]; then
			unset SLURM_JOB_ID SLURM_CPUS_PER_TASK
		else
			export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK="$cores"
		fi

		local start elapsed
		start=$(date +%s)
		$PY src/lead/training/train.py \
			"${common[@]}" \
			"training.data.loader_in_order=$in_order" \
			"training.experiment.output_dir=$OUT/$name" \
			> "$armlog" 2>&1
		local status=$?
		elapsed=$(( $(date +%s) - start ))

		if [ $status -ne 0 ]; then
			say "  arm $name FAILED (exit $status)"
			tail -5 "$armlog" | tee -a "$LOG"
			return 1
		fi

		# The trainer prints a running average; the last one is the settled
		# rate, after compilation has stopped costing anything.
		local rate
		rate=$(tr '\r' '\n' < "$armlog" \
			| grep -oE '[0-9.]+ samples/s' | tail -1)
		local peak
		peak=$(tr '\r' '\n' < "$armlog" \
			| grep -oE '[0-9.]+ GB peak GPU memory' | tail -1)
		say "  arm $name: ${rate:-no rate parsed}, ${peak:-no peak parsed}, wall ${elapsed}s"
	}

	arm baseline_8 default true
	arm workers_24 24 true
	arm workers_24_unordered 24 false

	say "done. per-arm logs in ~/bench_*.log"
	say "scratch checkpoints under $OUT -- safe to delete"
	unset SLURM_JOB_ID SLURM_CPUS_PER_TASK
}

main
