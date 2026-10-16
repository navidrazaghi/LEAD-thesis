#!/bin/bash
#
# The two attribution runs, in the order that survives running out of time.
#
# By the time this starts, the recipe-era ladder has three rungs and one gap
# each side of the largest result:
#
#     post31          dense baseline
#     rung2a_recipe   + degradation curriculum          15.84 -> 31.07 lidar
#     rung2ad_recipe  + deformable and calibrated refs  31.07 -> 59.74 lidar
#     rung3_recipe    + observability head and gate
#
# Two of those arrows carry more than one change, and this closes both.
#
# rung1d first. The +29 under LiDAR damage is the project's largest measured
# gain and it currently belongs to a package -- the deformable operator and
# the calibrated reference points together. The ladder's open-loop table says
# the split matters and says it awkwardly: geometry-free beat calibrated in all
# three columns, and only with the curriculum did calibrated become best. The
# calibrated flag adds exactly zero parameters, so this is as clean an
# isolation as an experiment gets, and it is asking about a claimed
# contribution rather than about a negative result. If time runs out, this is
# the one that had to happen.
#
# The observability mask second, in the slot rung 2b used to hold. That slot
# was reserved on a condition this file wrote down: rung 2b separates the head
# from the gate, and its value depends on rung 3 having something to attribute,
# so if rung 3 landed level with rung2ad_recipe there would be no deficit to
# split. Rung 3 landed level -- +2.48, -4.81 and -4.08 driving score, every one
# inside its own standard error -- so the condition came true and the slot goes
# to the run that can still produce something.
#
# The mask keeps the head and changes what is done with its signal: where the
# head doubts a token own modality, the token moves towards a learned prior
# instead of the operator merely reading less from it. Reweighting leaves the
# damaged features in the branch trunk that feeds the planning decoder, and has
# nothing to offer when both modalities are degraded, which is the weather case
# the proposal set out to improve.
#
# Only one of the two fits before the machine expires. Splitting a null is
# worth less than one more chance at a result.
#
# If the first run fails, the second still runs.

set -u

say() { echo "[$(date '+%m-%d %H:%M:%S')] chain3: $*"; }

say "run 1 of 2: rung 1d -- deformable, geometry-free, curriculum"
bash "$HOME/run_rung1d_recipe.sh" >> "$HOME/rung1d_recipe_driver.log" 2>&1
first=$?
say "run 1 exited $first"

say "run 2 of 2: observability mask -- substitution instead of a bias"
bash "$HOME/run_mask_recipe.sh" >> "$HOME/mask_recipe_driver.log" 2>&1
second=$?
say "run 2 exited $second"

REPO="$HOME/LEAD/lead"
TRACKED="$REPO/thesis-artifacts/results"
carried=""
say "results:"
for name in closed_loop_rung1d_recipe.csv closed_loop_mask_recipe.csv; do
	csv="$REPO/results/$name"
	if [ -s "$csv" ]; then
		say "  $(( $(wc -l < "$csv") - 1 )) rows in $name"
		cp "$csv" "$TRACKED/$name"
		carried="$carried thesis-artifacts/results/$name"
	else
		say "  MISSING $name"
	fi
done

if [ -n "$carried" ]; then
	# shellcheck disable=SC2086
	if git -C "$REPO" add $carried &&
		git -C "$REPO" commit -q -m "Carry the two attribution results out of the machine

Written by run_final_chain3.sh when the runs finished. results/ is ignored, so
until this commit these rows existed on one disk only." -- $carried; then
		say "committed$carried"
		say "NOT pushed: git -C ~/LEAD/lead push thesis robust-deployment"
	else
		say "commit failed or nothing changed; the files are in $TRACKED either way"
	fi
fi

say "the recipe-era ladder is now every component with a one-flag neighbour:"
say "  post31 -> rung2a -> rung1d -> rung2ad -> rung3, and the mask beside it"
say "  curriculum, operator, calibration, head, gate."
exit $(( first != 0 || second != 0 ))
