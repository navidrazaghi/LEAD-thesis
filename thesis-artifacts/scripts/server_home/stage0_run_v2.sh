#!/bin/bash
#
# Stage 0: merge the fault catalogue, prove the old corruptions intact, then find
# where the seed-2 model breaks.
#
# The catalogue was built in a separate worktree so the seed-2 evaluation could
# not import it half-finished. This waits until everything seed 2 queued has
# finished, then:
#
#   1. copies the catalogue into the main checkout -- but only if the three
#      existing files it touches are still exactly as committed there, so no
#      change made in the meantime is overwritten;
#   2. re-runs the byte-identity check on the main checkout and compares it with
#      the hashes taken before the merge. Every old corruption path (camera:1.0,
#      lidar:1.0, occlusion, ego_state, curriculum v1 and v2) must produce the
#      same bytes, or nothing is evaluated;
#   3. runs the catalogue's tests in the main checkout;
#   4. drives two checkpoints under all 13 faults at severities 0.5 and 1.0 on
#      10 of the 30 routes -- every third, so the subset is fixed in advance
#      rather than chosen by a result: the seed-2 consistency model (585 logs)
#      and the published LEAD reference (the 1.1 TB release). Each fault is
#      summarised against the same model intact on the same routes, and the two
#      models' drops are compared route by route.
#
# Body in a function called on the last line.

main() {
	set -u
	B=/home/new_drive/razaghi
	MAIN=$B/lead
	WT=$B/lead_stage0
	PY=$B/miniforge3/envs/lead/bin/python
	CKPT=$MAIN/outputs/rung2a_diverse_consistency_seed2_post31
	REF=$MAIN/reference/resnet34_v150_seed0
	ROUTES=$B/fault_probe_routes.txt
	CSV=results/closed_loop_fault_catalog_probe.csv
	FILES_CHANGED="src/lead/policy/transfuser/utils/sensor_degradation.py src/lead/policy/transfuser/transfuser.py scripts/common/run_evaluation.py"
	FILES_NEW="src/lead/policy/transfuser/utils/fault_catalog.py tests/unittests/policy/transfuser/test_fault_catalog.py"

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	say "waiting for the seed-2 chain and its camera replicate"
	while pgrep -f '[r]un_diverse_dense_consistency_seed2' > /dev/null || pgrep -f '[s]eed2_camera_rep' > /dev/null \
		|| pgrep -f '[l]ead.training.train' > /dev/null || pgrep -f '[e]val_parallel' > /dev/null; do
		sleep 120
	done
	[ -f "$CKPT/model_0030.pth" ] || { say "FATAL: no seed-2 checkpoint"; exit 1; }
	[ -f "$REF/model_0030.pth" ] || { say "FATAL: no reference checkpoint"; exit 1; }

	# --- 1. merge ------------------------------------------------------------
	cd "$MAIN" || exit 1
	if ! git diff --quiet -- $FILES_CHANGED; then
		say "FATAL: the main checkout's copies of the touched files changed since the worktree branched; not overwriting"
		git diff --stat -- $FILES_CHANGED
		exit 1
	fi
	for f in $FILES_CHANGED $FILES_NEW; do
		mkdir -p "$(dirname "$MAIN/$f")"
		cp "$WT/$f" "$MAIN/$f"
	done
	say "catalogue merged: $(git diff --shortstat) plus 2 new files"

	# --- 2. the old corruptions, byte for byte --------------------------------
	cd /tmp || exit 1
	PYTHONPATH=$MAIN/src CUDA_VISIBLE_DEVICES="" TIMM_USE_OLD_CACHE=1 $PY $B/old_paths_identity.py 2>/dev/null \
		| grep -vE "Warning|warn" > /tmp/identity_after_merge.txt
	if diff <(tail -n +2 $B/identity_before_merge.txt) <(tail -n +2 /tmp/identity_after_merge.txt) > /dev/null; then
		say "old corruption paths byte-identical after the merge ($(( $(wc -l < /tmp/identity_after_merge.txt) - 1 )) paths)"
	else
		say "FATAL: an old corruption path changed after the merge:"
		diff <(tail -n +2 $B/identity_before_merge.txt) <(tail -n +2 /tmp/identity_after_merge.txt)
		exit 1
	fi

	# --- 3. tests in the main checkout ----------------------------------------
	cd "$MAIN" || exit 1
	if ! PYTHONPATH=$MAIN/src CUDA_VISIBLE_DEVICES="" TIMM_USE_OLD_CACHE=1 timeout 900 $PY -m pytest -q -p no:cacheprovider \
		tests/unittests/policy/transfuser/test_fault_catalog.py \
		tests/unittests/policy/transfuser/test_degrade_batch_family.py \
		tests/unittests/policy/transfuser/test_sensor_degradation.py > /tmp/stage0_tests.txt 2>&1; then
		say "FATAL: tests failed in the main checkout"; tail -15 /tmp/stage0_tests.txt; exit 1
	fi
	say "tests: $(tail -1 /tmp/stage0_tests.txt)"

	# --- 4. the collapse sweep --------------------------------------------------
	awk 'NR % 3 == 1' src/lead/routes/eval_sets/degradation_30.txt > "$ROUTES"
	CONDITIONS=""
	for fault in cam_signal_drop cam_local_noise cam_exposure_pulse cam_local_occlusion cam_night_lowlight \
		cam_motion_blur cam_ghosting cam_color_shift lid_signal_drop lid_range_dropout lid_frustum_occlusion \
		lid_local_speckle lid_feature_noise; do
		CONDITIONS="$CONDITIONS $fault:0.5 $fault:1.0"
	done
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LD_LIBRARY_PATH="$B/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	ulimit -n 65536
	say "sweep: $(wc -l < "$ROUTES") routes x 26 conditions x 2 checkpoints, 4 shards"
	$PY thesis-artifacts/scripts/server_home/eval_parallel_v2.py \
		--models dense_consistency_seed2="$CKPT" reference_v150="$REF" \
		--routes "$ROUTES" \
		--conditions $CONDITIONS \
		--out "$CSV" --shards 4 \
		|| say "WARNING: the sweep reported missing rows; see outputs/eval_shards"
	$PY $B/fault_probe_summary.py "$CSV" "$ROUTES"
	say "stage 0 sweep finished"
}

main "$@"
