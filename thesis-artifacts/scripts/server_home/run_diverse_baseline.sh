#!/bin/bash
#
# The unmodified LEAD baseline, retrained on the diverse subset.
#
# Same model and recipe as rung0_lead_recipe + rung0_lead_recipe_post31 (the
# 43.20 baseline): TransFuser with every thesis addition off, 31 + 31 epochs,
# batch 32 x accumulation 2. The one intended difference is the data:
# ~/new_subset/selected_frames_town.txt -- 585 logs, 12 towns, 43 scenario
# types, the same 312k frames as the old 450. And, now that the perturbated
# view is on disk, the recipe's own use_sensor_perturbation (already true in
# the old config) finally fires; the old runs silently fell back to normal
# views only.
#
# Steps: wait for the download -> build the cache (both views) -> pretrain ->
# post-train -> 30 routes x 3 conditions. Each step refuses to start if the
# previous one did not finish cleanly.
#
# Body in a function called on the last line, so editing this file while it
# runs cannot feed bash shifted bytes.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung0_diverse
	POST=$HOME/LEAD/lead/outputs/rung0_diverse_post31
	CSV=results/closed_loop_diverse_baseline.csv
	LOG=$HOME/diverse_baseline.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	# 16 loader / cache workers: the measured loader optimum (556 samples/s,
	# against 346 at the old 8). runtime_variables reads the core count from
	# SLURM only; outside SLURM it falls back to 8.
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	# --- 1. download ---------------------------------------------------------
	while pgrep -f '[f]etch_selected.py' > /dev/null; do sleep 60; done
	if ! grep -q '^done, 0 failed' ~/bulk_download.log; then
		say "FATAL: download did not finish cleanly:"; tail -5 ~/bulk_download.log
		exit 1
	fi
	NAMES=$(awk -F/ '{print $2}' "$SEL" | sort)
	N=$(echo "$NAMES" | wc -l)
	MISSING=0
	for view in normal_view perturbated_view; do
		for s in $(cat "$SEL"); do
			[ -f "data/lead/123D/logs/$view/$s/sync.arrow" ] || MISSING=$((MISSING + 1))
		done
	done
	if [ "$N" -ne 585 ] || [ "$MISSING" -ne 0 ]; then
		say "FATAL: $N logs selected, $MISSING log views without sync.arrow"
		exit 1
	fi
	LOG_NAMES=$(echo "$NAMES" | paste -sd, -)
	say "download verified: $N logs, both views"

	# --- 2. cache --------------------------------------------------------------
	# No forced rebuild: logs already in the store are skipped, and the store
	# checks its fingerprint before any work. build_cache ignores
	# py123d_log_names and caches every log on disk, so the 28 held-out logs get
	# cached too; that is harmless, because training below pins the 585 names.
	say "building the cache, 16 workers"
	$PY -m lead.training.build_cache \
		training.data.force_cache_rebuild=false \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		>> "$LOG" 2>&1 || { say "FATAL: cache build failed"; tail -20 "$LOG"; exit 1; }
	say "cache built"

	COMMON=(
		training.data.read_from_cache_store=true
		"training.data.py123d_log_names=[$LOG_NAMES]"
		training.optimization.batch_size=32
		training.lightning.accumulate_grad_batches=2
		training.optimization.num_epochs=31
	)

	# --- 3. pretrain -------------------------------------------------------
	if [ ! -f "$PRE/model_0030.pth" ]; then
		say "pretrain starting: 31 epochs"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" \
			>> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"

	# --- 4. post-train -----------------------------------------------------
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

	# --- 5. evaluation -----------------------------------------------------
	say "scoring 30 routes x 3 conditions"
	$PY scripts/common/run_evaluation.py \
		--models diverse="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" \
		>> "$LOG" 2>&1
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "compare with results/closed_loop_post31.csv: same model and recipe, different data"
}

main "$@"
