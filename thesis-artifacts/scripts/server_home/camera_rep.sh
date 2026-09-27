#!/bin/bash
#
# A second run of the camera-destroyed condition on this machine.
#
# Test B paired the published checkpoint across the two servers. Intact and
# LiDAR-destroyed agreed -- 24/30 and 17/27 routes scored identically -- but
# camera-destroyed did not: 6/25 identical and a mean 8.2 lower (SE 5.5, 95%
# interval -19.6 to +3.1). Two explanations fit and that run cannot tell them
# apart: driving on LiDAR alone amplifies run-to-run noise, or something in the
# camera damage path differs between the machines (it has a blur that runs as a
# bfloat16 convolution; the LiDAR damage is a plain dropout).
#
# A second run here separates them. If new-vs-new differs as much as old-vs-new,
# it is noise. If the two new runs agree and both sit away from the old one, the
# machines differ.
#
# Everything is held the same as test B -- the same eval_parallel.py, three
# shards, the same checkpoint and routes; the damage is seeded per route, so
# both runs meet identical noise. Only the model label changes, so the results
# file does not skip the rows test B already wrote.
#
# Waits for the speed test, which needs the GPU to itself.
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

	say "waiting for the speed test"
	while pgrep -f '[s]peed_test.sh' > /dev/null || pgrep -f '[t]est_a2.sh' > /dev/null \
		|| pgrep -f '[l]ead.training.train' > /dev/null; do
		sleep 60
	done

	say "camera:1.0 again, 30 routes, 3 shards, same tool as test B"
	$PY thesis-artifacts/scripts/server_home/eval_parallel.py \
		--models reference_v150_rep2=/home/new_drive/razaghi/lead/reference/resnet34_v150_seed0 \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions camera:1.0 \
		--out results/closed_loop_reference_v150_camera_rep2.csv \
		--shards 3
	say "run finished"
	$PY /home/new_drive/razaghi/rep_compare.py
	say "camera replicate finished"
}

main "$@"
