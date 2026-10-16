#!/bin/bash
#
# Start the weather campaign when the two-run chain is finished.
#
# WHY A SECOND WAITER RATHER THAN AN EDIT
#
# run_final_chain.sh is running right now. Bash reads a script by byte offset
# as it executes, so appending a third step to a file already in flight feeds
# the running shell garbage from the point the offsets shift. Every script in
# this project carries that warning in its header; this is the case it warns
# about. So the chain is left untouched and this waits behind it instead.
#
# WHY IT WAITS ON THE CHAIN AND NOT ON THE GPU
#
# The same reason run_after_current.sh does. run_evaluation.py drives CARLA
# route by route and the card can be momentarily free between routes, so a
# GPU-only check can call it idle while ninety routes are still pending. The
# chain's own process is alive for exactly as long as the work is, which is
# the signal that cannot lie.
#
# It starts the campaign whatever the chain exited with. A failed rung 3 costs
# the campaign one of its five models and the run script says so and carries
# on; it is not a reason to leave the card idle for two days.

set -u

say() { echo "[$(date '+%m-%d %H:%M:%S')] waiter2: $*"; }

CHAIN_PATTERN="run_final_chain[.]sh"
EVAL_PATTERN="scripts/common/run_evaluation[.]py"

say "waiting for the two-run chain to finish"

while pgrep -f "$CHAIN_PATTERN" > /dev/null; do
	sleep 300
done
say "the chain has exited"

# Belt and braces: the chain could exit while a child evaluation winds down.
while pgrep -f "$EVAL_PATTERN" > /dev/null; do
	sleep 120
done
say "no evaluation is running"

for name in closed_loop_rung2ad_recipe closed_loop_rung3_recipe; do
	csv="$HOME/LEAD/lead/results/$name.csv"
	if [ -s "$csv" ]; then
		say "  $(( $(wc -l < "$csv") - 1 )) of 90 rows in $name.csv"
	else
		say "  WARNING: $name.csv is missing or empty"
	fi
done

say "starting run_weather_recipe.sh"
exec bash "$HOME/run_weather_recipe.sh"
