#!/bin/bash
#
# After stage 0 merges the fault catalogue into the main checkout, run the whole
# unit suite there and compare it with the suite as it stood before the merge.
#
# In the worktree two test files could not be collected (their `leaderboard`
# import resolves only from the main checkout), so the worktree run could not
# vouch for them. Run from the main checkout after the merge, every file is
# collected, and any test failing now that passed before is the merge's doing.
#
# Body in a function called on the last line.

main() {
	set -u
	B=/home/new_drive/razaghi
	MAIN=$B/lead
	C=$B/CARLA/standard_0916/PythonAPI/carla
	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	say "waiting for stage 0 to merge the catalogue"
	until grep -q "catalogue merged" $B/stage0_run.log 2>/dev/null; do
		pgrep -f '^bash stage0_run_v2.sh' > /dev/null || { say "stage 0 ended without merging; nothing to test"; exit 0; }
		sleep 120
	done
	sleep 60
	cd "$MAIN" || exit 1
	PYTHONPATH=$MAIN/src:$C CUDA_VISIBLE_DEVICES="" TIMM_USE_OLD_CACHE=1 timeout 3000 \
		$B/miniforge3/envs/lead/bin/python -m pytest -q -p no:cacheprovider --continue-on-collection-errors tests/unittests 2>&1 \
		| grep -E "^(FAILED|ERROR) |[0-9]+ (passed|failed)" > $B/full_suite_after_merge.txt
	say "before merge: $(grep -E '[0-9]+ (passed|failed)' $B/full_suite_before_merge.txt)"
	say "after merge:  $(grep -E '[0-9]+ (passed|failed)' $B/full_suite_after_merge.txt)"
	NEW=$(comm -23 <(grep -E "^(FAILED|ERROR) " $B/full_suite_after_merge.txt | sed "s/ - .*//" | sort) \
		<(grep -E "^(FAILED|ERROR) " $B/full_suite_before_merge.txt | sed "s/ - .*//" | sort))
	if [ -z "$NEW" ]; then
		say "no test fails after the merge that passed before it"
	else
		say "REGRESSION: failing only after the merge:"
		echo "$NEW"
	fi
}

main "$@"
