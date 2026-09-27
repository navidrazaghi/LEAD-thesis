#!/bin/bash
#
# Test A, done right: the campaign's second stage, then drive what it writes.
#
# The first test A drove the smoke pretrain checkpoint and it crashed on both
# routes. That checkpoint has use_planning_decoder: false -- it is stage one of
# the recipe, perception pretraining -- while the agent takes its controls from
# the planning decoder. The campaign never evaluated a pretrain checkpoint;
# every model was post-trained with use_planning_decoder=true first. So this
# runs that post-train for a few batches from the smoke pretrain, exactly as
# run_diverse_baseline.sh does it, and drives the result. That tests the second
# training stage, which has not run on this machine either.
#
# Waits for test B: post-training takes about 20 GB of the GPU, and three CARLA
# shards hold 25 of the 40.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	SEL=thesis-artifacts/provenance/new_subset/selected_frames_town.txt
	PRE=/home/new_drive/razaghi/lead/outputs/smoke
	POST=/home/new_drive/razaghi/lead/outputs/smoke_post
	LOG=/home/new_drive/razaghi/test_a2_train.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export LD_LIBRARY_PATH="/home/new_drive/razaghi/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	say "waiting for test B to finish"
	while pgrep -f '[s]ystem_tests.sh' > /dev/null || pgrep -f '[e]val_parallel.py' > /dev/null; do
		sleep 60
	done

	LOG_NAMES=$(awk -F/ '{print $2}' "$SEL" | sort | paste -sd, -)

	# --- post-train, as run_diverse_baseline.sh step 4, a few batches --------
	say "post-train: 2 epochs x 30 batches from $PRE/model_0001.pth, planning decoder on"
	rm -rf "$POST"
	$PY -m lead.training.train \
		training.data.read_from_cache_store=true \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		training.optimization.batch_size=32 \
		training.lightning.accumulate_grad_batches=2 \
		training.optimization.num_epochs=2 \
		training.lightning.limit_train_batches=30 \
		policy.transfuser.use_planning_decoder=true \
		training.experiment.resume_from_last_checkpoint=false \
		training.experiment.initial_weights_file="$PRE/model_0001.pth" \
		training.experiment.output_dir="$POST" \
		> "$LOG" 2>&1
	if ! ls "$POST"/model_*.pth > /dev/null 2>&1; then
		say "FATAL: post-train wrote no checkpoint"; tail -30 "$LOG"; exit 1
	fi
	say "post-train wrote $(ls "$POST"/model_*.pth | xargs -n1 basename | tr '\n' ' ')"
	grep -n "use_planning_decoder" "$POST/config.yaml" | head -1

	# --- drive it --------------------------------------------------------------
	say "driving the post-trained smoke checkpoint on 2 routes"
	$PY scripts/common/run_evaluation.py \
		--models smoke_post="$POST" \
		--routes /tmp/smoke_routes.txt \
		--conditions none:0 \
		--port 3100 \
		--work-dir outputs/eval_scratch_smoke \
		--out results/system_test_smoke_load.csv
	$PY /home/new_drive/razaghi/old_vs_new.py 2>/dev/null | sed -n '1,/test B/p'
	say "test A2 finished"
}

main "$@"
