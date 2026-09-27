#!/bin/bash
#
# A second camera-destroyed evaluation of the seed-2 consistency model.
#
# Seeds 0 and 1 of this model differed by 14.4 points with the camera destroyed,
# and seed_replicates_record.md could not say how much of that was training
# variability and how much the simulator's run-to-run variability. The reference
# checkpoint, driven twice on this machine, already put the second at 5.8 points
# on the mean and 12.5 per route in this condition. Two evaluations of one
# trained model -- this one -- measure it for the model actually in question.
#
# Held identical to the seed-2 chain's own scoring: eval_parallel_v2.py, four
# shards, the same routes, and damage seeded per route. Only the label differs,
# so the results file does not skip the rows the first evaluation wrote.
#
# Waits for the seed-2 chain to finish training and its first evaluation.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	POST=$HOME/LEAD/lead/outputs/rung2a_diverse_consistency_seed2_post31
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LD_LIBRARY_PATH="/home/new_drive/razaghi/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	ulimit -n 65536

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	say "waiting for the seed-2 chain"
	while pgrep -f '[r]un_diverse_dense_consistency_seed2' > /dev/null \
		|| pgrep -f '[l]ead.training.train' > /dev/null || pgrep -f '[e]val_parallel' > /dev/null; do
		sleep 120
	done
	[ -f "$POST/model_0030.pth" ] || { say "FATAL: no seed-2 post-train checkpoint; the chain did not finish"; exit 1; }

	say "camera:1.0, second evaluation, 30 routes, 4 shards"
	$PY thesis-artifacts/scripts/server_home/eval_parallel_v2.py \
		--models dense_consistency_seed2_rep2="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions camera:1.0 \
		--out results/closed_loop_diverse_dense_consistency_seed2_camera_rep2.csv \
		--shards 4
	$PY /home/new_drive/razaghi/seed2_camera_compare.py
	say "seed-2 camera replicate finished"
}

main "$@"
