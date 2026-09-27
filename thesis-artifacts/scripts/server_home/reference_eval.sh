#!/bin/bash
#
# Score the published LEAD checkpoint on our own 30 routes.
#
# This is a test of the evaluation harness, not of the model. resnet34_v1.5.0
# was trained on the whole 1.1 TB release, normal and perturbated view, on four
# H100s, and its published Bench2Drive score is 93.6 +/- 1.0 DS. If our harness
# drives it round the same routes and reports something in that neighbourhood,
# the harness is sound and the campaign's numbers mean what they say. If it
# reports 20, the fault is ours and every comparison built on this pipeline is
# suspect.
#
# Two things keep the comparison honest. The published 93.6 is over all 220
# Bench2Drive routes; ours is the 30-route degradation set, a harder and much
# smaller sample, so the two numbers are related but not the same measurement.
# And the checkpoint ships a 1.5.0 config while this repo is 1.4.0 with the
# thesis changes on top, so it may simply refuse to load -- which is itself
# worth knowing early.
#
# Runs on the intact condition only. The download costs 276 MB and needs no
# dataset: closed-loop evaluation drives in CARLA.
#
# Body in a function called on the last line.

main() {
	set -u
	BASE=/home/new_drive/razaghi
	REPO=$BASE/lead
	PY=$BASE/miniforge3/envs/lead/bin/python
	REF=$REPO/reference/resnet34_v150_seed0
	CARLA=$BASE/CARLA/standard_0916
	CSV=$REPO/results/closed_loop_reference_v150.csv
	LOG=$BASE/reference_eval.log
	SRC=https://hf-mirror.com/ln2697/transfuser-carla-123d/resolve/main/resnet34_v1.5.0/seed0

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	mkdir -p "$REF" "$REPO/results"

	# --- 1. the checkpoint ---------------------------------------------------
	for file in config.yaml model_0030.pth; do
		if [ -s "$REF/$file" ]; then
			say "$file already here: $(du -h "$REF/$file" | cut -f1)"
			continue
		fi
		say "downloading $file"
		for attempt in 1 2 3 4 5 6 7 8 9 10; do
			# -L for the same reason it mattered in the dataset fetch: without
			# it curl writes the redirect body and exits 0.
			curl -4 -fsSL -C - --connect-timeout 30 --max-time 3600 \
				--speed-limit 2000 --speed-time 60 \
				-o "$REF/$file" "$SRC/$file" && break
			say "  attempt $attempt dropped; retrying"
			sleep 30
		done
		[ -s "$REF/$file" ] || { say "FATAL: $file did not download"; exit 1; }
		say "$file: $(du -h "$REF/$file" | cut -f1)"
	done

	# --- 2. drive it ---------------------------------------------------------
	cd "$REPO" || exit 1
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	ulimit -n 65536

	say "scoring 30 routes, intact condition"
	$PY scripts/common/run_evaluation.py \
		--models reference_v150="$REF" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 \
		--carla-root "$CARLA" \
		--port 3100 \
		--work-dir "$REPO/outputs/eval_scratch_reference" \
		--out "$CSV" \
		>> "$LOG" 2>&1

	if [ -f "$CSV" ]; then
		say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
		say "published Bench2Drive score for this checkpoint: 93.6 +/- 1.0 DS over 220 routes"
	else
		say "FATAL: no results CSV; last lines of $LOG:"
		tail -30 "$LOG"
		exit 1
	fi
}

main "$@"
