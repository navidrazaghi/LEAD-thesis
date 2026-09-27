#!/bin/bash
#
# Two tests of the evaluation path on this machine, before any long training.
#
# Test A: a checkpoint written by this machine's own training loads and drives.
# The only model the harness has driven here so far is the published one, whose
# config.yaml comes from lead 1.5.0. The smoke run's checkpoint is two epochs
# old, so its score means nothing; what is tested is that the agent sets up
# from a config and weights this pipeline produced, and moves the car.
#
# Test B: the published checkpoint under the two conditions the thesis leans
# on, LiDAR destroyed and camera destroyed. The sensor-degradation code has not
# run once on this machine. The old server drove this same checkpoint on these
# same 30 routes under both conditions (thesis-artifacts/results/
# reference_closed_loop.csv), so the two pair route by route.
#
# A runs first and alone: four CARLA instances on 16 cores would slow every
# route, and a route that runs into its time limit is scored as the model's
# failure rather than the machine's.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LD_LIBRARY_PATH="/home/new_drive/razaghi/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	ulimit -n 65536

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	command -v ffmpeg > /dev/null || { say "FATAL: ffmpeg not on PATH"; exit 1; }

	# --- A. our own checkpoint -------------------------------------------------
	printf '25968.xml\n26956.xml\n' > /tmp/smoke_routes.txt
	say "test A: the smoke checkpoint on 2 routes, intact"
	$PY scripts/common/run_evaluation.py \
		--models smoke=/home/new_drive/razaghi/lead/outputs/smoke \
		--routes /tmp/smoke_routes.txt \
		--conditions none:0 \
		--port 3100 \
		--work-dir outputs/eval_scratch_smoke \
		--out results/system_test_smoke_load.csv
	say "test A done"

	# --- B. the published checkpoint, sensors destroyed ------------------------
	say "test B: reference_v150 on 30 routes x {lidar:1.0, camera:1.0}, 3 shards"
	$PY thesis-artifacts/scripts/server_home/eval_parallel.py \
		--models reference_v150=/home/new_drive/razaghi/lead/reference/resnet34_v150_seed0 \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions lidar:1.0 camera:1.0 \
		--out results/closed_loop_reference_v150.csv \
		--shards 3
	say "test B done"

	$PY /home/new_drive/razaghi/old_vs_new.py
	say "all tests finished"
}

main "$@"
