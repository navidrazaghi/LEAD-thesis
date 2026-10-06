#!/bin/bash
#
# Start the last two runs as soon as the corrected rung 2a finishes scoring.
#
# WHY NOT JUST WAIT FOR THE GPU
#
# Each run script waits for our own processes to leave the card before it
# starts, and that is the right guard between two trainings. It is the wrong
# guard behind an evaluation. run_evaluation.py drives CARLA route by route,
# and the card can be momentarily free between routes -- long enough for a
# GPU-only check to call it idle and start a training on top of a campaign
# that still has fifty routes to go. Both would then be wrong: the training
# would be slow, and the evaluation would start losing routes to the harness
# timeout, which is exactly how the intact column of
# closed_loop_new_baseline.csv was ruined and had to be rescored.
#
# So this waits on the driver, which exists for as long as the run does, and
# on run_evaluation.py after it, the way run_intact_rerun.sh already does.
# Only when both are gone does it hand over to the chain, whose own GPU guard
# then handles whatever CARLA left behind.
#
# WHAT IT STARTS
#
# run_final_chain.sh: rung 2a deformable at the published recipe first, then
# rung 3. That order is deliberate and is argued in the chain script.
#
# It starts the chain whatever the evaluation produced. A short CSV is a
# reason to read the log, not a reason to leave the card idle overnight; the
# count is printed here so the morning starts with it.

set -u

say() { echo "[$(date '+%m-%d %H:%M:%S')] waiter: $*"; }

DRIVER_PATTERN="run_rung2a_recipe[.]sh"
EVAL_PATTERN="scripts/common/run_evaluation[.]py"
CSV="$HOME/LEAD/lead/results/closed_loop_rung2a_recipe.csv"

say "waiting for the corrected rung 2a to finish training and scoring"

while pgrep -f "$DRIVER_PATTERN" > /dev/null; do
	sleep 300
done
say "the driver has exited"

# Belt and braces: the driver could in principle exit while a child evaluation
# is still winding down.
while pgrep -f "$EVAL_PATTERN" > /dev/null; do
	sleep 120
done
say "no evaluation is running"

if [ -s "$CSV" ]; then
	say "$(( $(wc -l < "$CSV") - 1 )) of 90 rows in $(basename "$CSV")"
else
	say "WARNING: $CSV is missing or empty; starting the chain anyway"
fi

say "starting run_final_chain.sh"
exec bash "$HOME/run_final_chain.sh"
