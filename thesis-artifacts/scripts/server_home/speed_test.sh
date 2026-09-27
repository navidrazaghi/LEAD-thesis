#!/bin/bash
#
# Training speed on this machine: compile mode default against max-autotune.
#
# The campaign's pretrain and post-train each took about 5.7 to 6 hours on the
# old server, about 87 samples/s. Two questions decide what that becomes here:
# whether the loader keeps up on 16 cores (the old machine had 32), and whether
# torch_compile_mode=max-autotune, which LEAD's docs/speed.md recommends for full
# runs, buys anything. max-autotune only picks different kernels; the recipe and
# the model are unchanged.
#
# Each training run is the campaign recipe exactly -- 585 pinned logs, batch 32
# with gradient accumulation 2, 16 loader workers, cache store -- stopped after
# 600 batches. Throughput is read between 25% and 100% of the run, so compile
# and autotune time are excluded from it and reported separately. Post-train
# starts from the smoke pretrain checkpoint, as the campaign's post-train starts
# from its pretrain.
#
# Waits for test A2, which waits for test B: nothing else may share the GPU while
# speed is being measured.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	B=/home/new_drive/razaghi
	SEL=thesis-artifacts/provenance/new_subset/selected_frames_town.txt
	OUT=$B/speed
	mkdir -p "$OUT"

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export LD_LIBRARY_PATH="$B/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	say "waiting for test A2 (and through it, test B)"
	while pgrep -f '[t]est_a2.sh' > /dev/null || pgrep -f '[s]ystem_tests.sh' > /dev/null \
		|| pgrep -f '[e]val_parallel' > /dev/null; do
		sleep 60
	done
	if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1000 ]; then
		say "FATAL: something still holds GPU memory; speed would be measured against it"
		nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
		exit 1
	fi

	# --- 1. the loader alone --------------------------------------------------
	say "loader alone, 8 and 16 workers"
	$PY $B/loader_bench.py > "$OUT/loader.log" 2>&1
	cat "$OUT/loader.log" | grep -vE "INFO|WARN" | tail -4

	LOG_NAMES=$(awk -F/ '{print $2}' "$SEL" | sort | paste -sd, -)

	run() {  # label, compile mode, extra dotlist tokens...
		local label=$1 mode=$2
		shift 2
		local dir=$OUT/run_$label
		rm -rf "$dir"
		say "training: $label"
		nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 2 > "$OUT/$label.gpu" &
		local gpu_pid=$!
		vmstat 5 > "$OUT/$label.vmstat" &
		local vm_pid=$!
		local t0=$(date +%s)
		$PY -m lead.training.train \
			training.data.read_from_cache_store=true \
			"training.data.py123d_log_names=[$LOG_NAMES]" \
			training.optimization.batch_size=32 \
			training.lightning.accumulate_grad_batches=2 \
			training.optimization.num_epochs=1 \
			training.lightning.limit_train_batches=600 \
			training.optimization.torch_compile_mode="$mode" \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.output_dir="$dir" \
			"$@" > "$OUT/$label.train" 2>&1
		local status=$?
		kill $gpu_pid $vm_pid 2>/dev/null
		say "  exit $status after $(( $(date +%s) - t0 )) s wall"
		$PY $B/speed_summary.py "$label" "$OUT/$label.train" "$OUT/$label.gpu" "$OUT/$label.vmstat"
		rm -rf "$dir"
	}

	# --- 2. pretrain, then post-train, in both compile modes ------------------
	run pretrain_default default
	run pretrain_autotune max-autotune
	run posttrain_default default \
		policy.transfuser.use_planning_decoder=true \
		training.experiment.initial_weights_file=$B/lead/outputs/smoke/model_0001.pth
	run posttrain_autotune max-autotune \
		policy.transfuser.use_planning_decoder=true \
		training.experiment.initial_weights_file=$B/lead/outputs/smoke/model_0001.pth

	say "reference: the campaign's pretrain ran at about 87 samples/s, post-train about 85, on the old server"
	say "speed test finished"
}

main "$@"
