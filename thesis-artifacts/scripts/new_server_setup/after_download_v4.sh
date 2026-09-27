#!/bin/bash
#
# Wait for the dataset, verify it, build the cache, then a short smoke training.
#
# The smoke run is the point of this script. No training has ever run on this
# machine, and the campaign runs are fifteen hours each; finding out there that
# the cache store, the loader's file limit or the CUDA stub is wrong would cost
# a day. So the same code path is exercised first on a few dozen batches: cache
# reads, forward, backward, an epoch boundary and a checkpoint write.
#
# The verification before it is the provenance's own: the 585 log directories it
# names must all be on disk with their sync.arrow. Only normal_view is checked.
# The campaign script checks both views because it was written expecting the
# perturbated view to be downloaded, but py123d_perturbated_split belongs to the
# expert's data-collection config, not the training loader, and no run that
# produced a thesis number ever read that view.
#
# Every step refuses to start if the one before it did not finish.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	REPO=$BASE/lead
	PY=$BASE/miniforge3/envs/lead/bin/python
	SEL=$REPO/thesis-artifacts/provenance/new_subset/selected_frames_town.txt
	SMOKE=$REPO/outputs/smoke
	LOG=$BASE/after_download.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	cd "$REPO" || exit 1

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	# 16 loader workers: the measured optimum on the old machine, 556 samples/s
	# against 346 at the default 8. runtime_variables reads the core count from
	# SLURM only, so both variables have to be set for it to leave 8 behind.
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	# pyturbojpeg==2.5 is pinned and needs libjpeg-turbo 3.x; Ubuntu 24.04 ships
	# 2.1.5. This directory holds only libturbojpeg.so.0 from libjpeg-turbo 3.2.0
	# (it links against libc and libm alone), so putting it first on the loader
	# path shadows nothing else. On 60 frames of this dataset it decodes
	# bit-identically to OpenCV's libjpeg-turbo 3.0.3 and to imagecodecs.
	export LD_LIBRARY_PATH="/home/new_drive/razaghi/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	ulimit -n 65536

	# --- 1. wait for the download -------------------------------------------
	# Also waits for the reference evaluation. The cache build runs sixteen
	# CPU workers for hours; alongside three CARLA shards on a 16-core machine
	# it would starve them, and a route that runs into its time limit is scored
	# as a failure of the model rather than of the machine.
	say "waiting for the completing download (depth and perturbated view)"
	while pgrep -f '[f]etch_stage4b.py' > /dev/null || pgrep -f '[e]val_parallel.py' > /dev/null; do
		sleep 120
	done
	if ! grep -q '^complete' "$BASE/stage4b.log"; then
		say "FATAL: the download did not report complete; last lines:"
		tail -5 "$BASE/stage4b.log"
		exit 1
	fi
	say "download reports complete"

	# --- 2. verify against the provenance list ------------------------------
	# The same check run_diverse_baseline.sh made before the campaign trained:
	# every log in both views. File counts too, since the first pass here was
	# short exactly by files the sync.arrow test cannot see (depth).
	N=$(grep -c . "$SEL")
	MISSING=0
	SHORT=0
	while read -r entry; do
		[ -n "$entry" ] || continue
		for view in normal_view perturbated_view; do
			[ -f "data/lead/123D/logs/$view/$entry/sync.arrow" ] || MISSING=$((MISSING + 1))
		done
		[ "$(ls "data/lead/123D/logs/normal_view/$entry" | wc -l)" -eq 19 ] || SHORT=$((SHORT + 1))
		[ "$(ls "data/lead/123D/logs/perturbated_view/$entry" | wc -l)" -eq 15 ] || SHORT=$((SHORT + 1))
	done < "$SEL"
	if [ "$N" -ne 585 ] || [ "$MISSING" -ne 0 ] || [ "$SHORT" -ne 0 ]; then
		say "FATAL: $N names, $MISSING log views without sync.arrow, $SHORT with the wrong file count"
		exit 1
	fi
	say "verified: all $N logs in both views, 19 + 15 files each"

	LOG_NAMES=$(awk -F/ '{print $2}' "$SEL" | sort | paste -sd, -)

	# --- 3. the cache -------------------------------------------------------
	# build_cache ignores py123d_log_names and caches every log on disk, so
	# naming them here only documents the intent; training pins them for real.
	# The cache built on the first pass saw neither depth nor the perturbated
	# view. It is derived data and cheap to rebuild, so it goes rather than
	# being trusted to notice what changed underneath it.
	rm -rf data/lead/123D/transfuser_training_cache
	say "building the cache from scratch, 16 workers"
	$PY -m lead.training.build_cache \
		training.data.force_cache_rebuild=false \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		>> "$LOG" 2>&1 || { say "FATAL: cache build failed"; tail -20 "$LOG"; exit 1; }
	say "cache built"

	# --- 4. the smoke training ----------------------------------------------
	# Two epochs of thirty batches: enough to cross an epoch boundary and write
	# a checkpoint, short enough to fail fast.
	say "smoke training: 2 epochs x 30 batches"
	rm -rf "$SMOKE"
	$PY -m lead.training.train \
		training.data.read_from_cache_store=true \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		training.optimization.batch_size=32 \
		training.optimization.num_epochs=2 \
		training.lightning.limit_train_batches=30 \
		training.experiment.resume_from_last_checkpoint=false \
		training.experiment.output_dir="$SMOKE" \
		>> "$LOG" 2>&1
	if ls "$SMOKE"/model_*.pth > /dev/null 2>&1; then
		say "smoke training passed; checkpoints written:"
		ls -la "$SMOKE"/model_*.pth | while read -r line; do say "  $line"; done
		say "the pipeline runs on this machine. Nothing further starts without approval."
	else
		say "FATAL: smoke training wrote no checkpoint; last lines of $LOG:"
		tail -30 "$LOG"
		exit 1
	fi
}

main "$@"
