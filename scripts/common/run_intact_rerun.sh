#!/bin/bash
# Score the retrained baseline's intact condition again, under a quieter machine.
#
# WHY
#
# The intact column of results/closed_loop_new_baseline.csv was collected while
# another user held seventeen of the thirty-two cores. Eleven of its thirty rows
# came back NoResult: the harness kills a route at _ROUTE_TIMEOUT_S of wall
# clock, and under that contention the cap bound before the routes could finish.
# Rows killed that way carry no score, and they are not missing at random -- a
# route is killed for taking too long, which is what a bad route does -- so the
# mean over the survivors is biased upward by an unknown amount. Manski bounds
# on that column run 12.99 to 49.65, which is wide enough to say nothing.
#
# The lidar column, collected after the processes were reniced and after a third
# user's job ended, has one censored row out of thirty and bounds of 16.52 to
# 19.86. Same protocol, same model, same routes: the difference is the machine.
#
# So this measures the intact condition again under the conditions the lidar
# column got, and nothing else changes.
#
# WHAT IS DELIBERATELY IDENTICAL
#
# The same thirty routes, the same model and checkpoint, the same condition
# string, the same harness. A re-run that changed any of those would answer a
# different question than the one being asked.
#
# WHAT IS DELIBERATELY SEPARATE
#
# Results go to their own file. The first attempt stays where it is: it is the
# evidence for what contention does to this protocol, and overwriting it would
# destroy the only paired comparison available -- same routes, same model, two
# machine loads. Reading them together is the point.
#
# ORDERING
#
# This waits for the running evaluation to exit rather than for the lock. The
# two would otherwise share a GPU and a CARLA port, and the second would slow
# the first while contaminating exactly the variable being measured.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local model="outputs/rung0_lead_recipe_post"
  local first="$HOME/LEAD/lead/results/closed_loop_new_baseline.csv"
  local out="$HOME/LEAD/lead/results/closed_loop_new_baseline_intact_rerun.csv"
  local routes="src/lead/routes/eval_sets/degradation_30.txt"

  echo "[$(date +%H:%M:%S)] waiting for the first evaluation to finish"
  while pgrep -f "scripts/common/run_evaluation[.]py" > /dev/null; do
    sleep 300
  done
  echo "[$(date +%H:%M:%S)] the first evaluation has exited"

  # It should have all ninety rows by then; if it stopped early, say so and
  # carry on, because this run is about the intact column either way.
  local rows
  rows=$(( $(wc -l < "$first" 2>/dev/null || echo 1) - 1 ))
  echo "[$(date +%H:%M:%S)] first file holds $rows of 90 rows"

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 604800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  echo "[$(date +%H:%M:%S)] scoring 30 routes, intact only"
  LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
    "$HOME/miniconda3/envs/lead/bin/python" scripts/common/run_evaluation.py \
      --models "rung0_lead=$model" \
      --routes "$routes" \
      --conditions none:0 \
      --out "$out"
  local code=$?

  echo "[$(date +%H:%M:%S)] evaluation exited $code"
  echo "[$(date +%H:%M:%S)] rows: $(( $(wc -l < "$out" 2>/dev/null || echo 1) - 1 )) of 30"
  echo
  echo "  Read this against the intact rows of the first file. Same routes, same"
  echo "  checkpoint, different machine load. If the censored count falls toward"
  echo "  the lidar column's one in thirty, the first column's width was the"
  echo "  machine and not the model."
  exit "$code"
}

main "$@"
