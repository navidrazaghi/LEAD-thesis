#!/bin/bash
#
# The last two runs, in the only order that survives running out of time.
#
# Rung 2a deformable first, then rung 3. Two reasons for that order, and both
# matter more than they look:
#
# 1. It is the run most likely to produce this thesis's best number. If the
#    machine goes away after one run, the one that happened is the one worth
#    having.
# 2. Rung 3 is only a two-flag comparison if rung 2a deformable exists to be
#    compared against. Run the other way round, rung 3 lands as a three-flag
#    jump and stays unattributable even after both have finished.
#
# Each script waits for our own processes to leave the GPU before it starts, so
# the second does not have to be timed against the first -- including the
# CARLA teardown at the end of an evaluation, which can outlive the harness.
#
# If the first run fails, the second still runs. Its result is worth less
# without the partner, and it says so in its own log, but a failure at 3am
# should not cost the whole night.
#
# THE RESULTS ARE CARRIED OUT AT THE END
#
# results/ is git-ignored, so a CSV written there exists on one disk and
# nowhere else. thesis-artifacts/results/ is the tracked copy, and every other
# result in this project reached the repository that way. Doing it here, in
# the script, means a machine that disappears the morning after the runs
# finish still leaves the numbers behind. The commit is local; pushing needs
# credentials this script must not have.

set -u

say() { echo "[$(date '+%m-%d %H:%M:%S')] chain: $*"; }

say "run 1 of 2: rung 2a deformable, published recipe"
bash "$HOME/run_rung2ad_recipe.sh" >> "$HOME/rung2ad_recipe_driver.log" 2>&1
first=$?
say "run 1 exited $first"

say "run 2 of 2: rung 3, published recipe"
bash "$HOME/run_rung3_recipe.sh" >> "$HOME/rung3_recipe_driver.log" 2>&1
second=$?
say "run 2 exited $second"

REPO="$HOME/LEAD/lead"
TRACKED="$REPO/thesis-artifacts/results"

say "results:"
carried=""
for name in closed_loop_rung2ad_recipe.csv closed_loop_rung3_recipe.csv; do
	csv="$REPO/results/$name"
	if [ -s "$csv" ]; then
		say "  $(( $(wc -l < "$csv") - 1 )) rows in $name"
		cp "$csv" "$TRACKED/$name"
		carried="$carried thesis-artifacts/results/$name"
	else
		say "  MISSING $name"
	fi
done

# Commit whatever arrived. A partial result is still a result, and a chain
# that ends without carrying one out is how a number gets lost.
if [ -n "$carried" ]; then
	# shellcheck disable=SC2086
	if git -C "$REPO" add $carried &&
		git -C "$REPO" commit -q -m "Carry the last two closed-loop results out of the machine

Written by run_final_chain.sh when the runs finished. results/ is ignored, so
until this commit these rows existed on one disk only." -- $carried; then
		say "committed$carried"
		say "NOT pushed: this script has no credentials. Push with"
		say "  git -C ~/LEAD/lead push thesis robust-deployment"
	else
		say "commit failed or nothing changed; the files are in $TRACKED either way"
	fi
fi

say "read both against results/closed_loop_rung2a_recipe.csv (dense, same"
say "recipe) and results/closed_loop_post31.csv (the corrected baseline)."
exit $(( first != 0 || second != 0 ))
