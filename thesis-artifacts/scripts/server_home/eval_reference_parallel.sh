#!/bin/bash
#
# The published LEAD checkpoint on our 30 routes, intact, three CARLA shards.
#
# The first attempt scored 0 on every route with "Agent couldn't be set up",
# which looked like a model incompatibility and was not: the agent refuses to
# start unless ffmpeg is on PATH, and this machine had none. ffmpeg now lives in
# its own conda env, kept out of the torch env, and is appended to PATH here.
#
# Three shards is what the campaign used, through this same eval_parallel.py.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	export PATH="$PATH:/home/new_drive/razaghi/miniforge3/envs/tools/bin"
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	ulimit -n 65536

	command -v ffmpeg > /dev/null || { echo "FATAL: ffmpeg not on PATH"; exit 1; }

	~/miniconda3/envs/lead/bin/python thesis-artifacts/scripts/server_home/eval_parallel.py \
		--models reference_v150=/home/new_drive/razaghi/lead/reference/resnet34_v150_seed0 \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 \
		--out results/closed_loop_reference_v150.csv \
		--shards 3
}

main "$@"
