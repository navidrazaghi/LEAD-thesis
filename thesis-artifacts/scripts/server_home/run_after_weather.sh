#!/bin/bash
#
# Start the two attribution runs when the weather campaign is finished.
#
# WHY IT WATCHES TWO NAMES
#
# run_after_chain.sh does not spawn the weather campaign, it execs into it, so
# one process changes its name partway through: run_after_chain.sh while it is
# still waiting, run_weather_recipe.sh once the campaign is running. Watching
# only the second would let this start during the wait; watching only the first
# would let it start on top of the campaign. Both names, one wait.
#
# WHY NOT AN EDIT TO THE EXISTING WAITER
#
# The same reason as before. run_after_chain.sh is already running and bash
# reads a script by byte offset as it executes, so appending to a file in
# flight feeds the running shell garbage. Each stage of this queue is added by
# putting a new waiter behind the last one, never by rewriting one that is
# already live.

set -u

say() { echo "[$(date '+%m-%d %H:%M:%S')] waiter3: $*"; }

WEATHER_PATTERN="run_after_chain[.]sh|run_weather_recipe[.]sh"
EVAL_PATTERN="scripts/common/run_evaluation[.]py"

say "waiting for the weather campaign to finish"

while pgrep -f "$WEATHER_PATTERN" > /dev/null; do
	sleep 300
done
say "the weather campaign has exited"

while pgrep -f "$EVAL_PATTERN" > /dev/null; do
	sleep 120
done
say "no evaluation is running"

csv="$HOME/LEAD/lead/results/weather_recipe.csv"
if [ -s "$csv" ]; then
	say "$(( $(wc -l < "$csv") - 1 )) rows in weather_recipe.csv"
else
	say "WARNING: weather_recipe.csv is missing or empty"
fi

say "starting run_final_chain3.sh"
exec bash "$HOME/run_final_chain3.sh"
