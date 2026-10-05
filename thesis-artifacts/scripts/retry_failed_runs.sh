#!/bin/bash
# Re-run the closed-loop cells that a simulator failure cost, then summarise.
#
# The sweep writes a row for every run it attempts, including the ones CARLA
# lost, and run_evaluation.py's load_done deliberately does not count those as
# measured -- so re-invoking it picks up exactly the failed cells and nothing
# else. What was missing is the second invocation: run_final_eval.sh calls the
# sweep once, and its pending list is computed before the first run starts.
#
# Why that matters more than a few missing numbers: every comparison in the
# thesis is paired by route, so a cell lost for one model drops that route from
# every comparison involving it. Losing cells unevenly across models is the
# same bias that the TickRuntime misclassification produced earlier, and it
# biases in an unpredictable direction because the routes that fail are not a
# random sample -- they are the hard ones.
#
# Passes are capped: a cell that fails twice is failing for a reason that
# another attempt will not fix, and each attempt can cost the full route
# timeout.

cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1

PY=~/miniconda3/envs/lead/bin/python
RES=~/LEAD/lead/results
CSV="$RES/closed_loop.csv"
MAX_PASSES=2

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Brackets keep pgrep from matching this script's own command line.
busy() {
  pgrep -f "run_final_[e]val.sh" > /dev/null 2>&1 \
    || pgrep -f "run_[e]valuation.py" > /dev/null 2>&1 \
    || pgrep -f "light_[e]xperiments.py" > /dev/null 2>&1 \
    || pgrep -f "Carla[U]E4-Linux" > /dev/null 2>&1
}

# The first version of this fired the moment the campaign was stopped by hand,
# and started a second sweep on top of the experiments that stop was made for.
# A gap in the process list is not proof the work has finished, so the
# condition has to hold twice, twenty minutes apart. That is longer than any
# deliberate pause taken so far and far shorter than the campaign.
log "waiting for the main sweep to finish"
while busy; do
  sleep 300
done
log "nothing running; confirming after a settling period"
sleep 1200
while busy; do
  sleep 300
done
log "the main sweep has exited"

if [ ! -f "$CSV" ]; then
  log "FAIL no results file at $CSV"
  exit 1
fi

# The models are whichever three the campaign actually measured, read from its
# own output rather than re-derived -- the third one was chosen at run time.
MODELS=$($PY - <<'PYEOF'
import csv
import pathlib

DIRS = {
    "rung0": "outputs/rung0_baseline_post",
    "rung2a": "outputs/rung2a_curriculum_only_post",
    "rung2b": "outputs/rung2b_observability_only_post",
    "rung3": "outputs/rung3_observability_gated_post",
    "rung4": "outputs/rung4_light_auxiliary_post",
}
path = pathlib.Path.home() / "LEAD/lead/results/closed_loop.csv"
seen = []
for row in csv.DictReader(path.open(encoding="utf-8")):
    name = row["model"]
    if name not in seen and name in DIRS:
        seen.append(name)
print(" ".join(f"{name}={DIRS[name]}" for name in seen))
PYEOF
)
if [ -z "$MODELS" ]; then
  log "FAIL could not work out which models were measured"
  exit 1
fi
log "models: $MODELS"

report() {
  $PY - <<'PYEOF'
import csv
import collections
import pathlib

path = pathlib.Path.home() / "LEAD/lead/results/closed_loop.csv"
rows = list(csv.DictReader(path.open(encoding="utf-8")))
good = collections.Counter()
lost = collections.Counter()
routes = collections.defaultdict(set)
for row in rows:
    key = (row["model"], f'{row["modality"]}:{row["severity"]}')
    if (row.get("driving_score") or "").strip():
        good[key] += 1
        routes[f'{row["modality"]}:{row["severity"]}'].add(row["route"])
    else:
        lost[key] += 1
print(f"  {'model':8}{'condition':14}{'measured':>10}{'lost':>7}")
for key in sorted(set(good) | set(lost)):
    print(f"  {key[0]:8}{key[1]:14}{good[key]:>10}{lost[key]:>7}")

# What the paired analysis can actually use: a route counts only where every
# model has a measurement for it.
print()
for condition in sorted(routes):
    per_model = collections.defaultdict(set)
    for row in rows:
        if f'{row["modality"]}:{row["severity"]}' != condition:
            continue
        if (row.get("driving_score") or "").strip():
            per_model[row["model"]].add(row["route"])
    if per_model:
        common = set.intersection(*per_model.values())
        print(f"  {condition:14} routes usable by every model: {len(common)}")
PYEOF
}

log "state before any retry:"
report

for pass_number in $(seq 1 $MAX_PASSES); do
  pending=$($PY scripts/common/run_evaluation.py \
    --models $MODELS \
    --routes src/lead/routes/eval_sets/degradation_30.txt \
    --conditions none:0 lidar:1.0 camera:1.0 \
    --out "$CSV" --plan-only 2>/dev/null | grep -oP '\d+(?= to go)')
  pending=${pending:-0}
  log "pass $pass_number: $pending cell(s) still unmeasured"
  if [ "$pending" -eq 0 ]; then
    log "nothing left to retry"
    break
  fi
  $PY scripts/common/run_evaluation.py \
    --models $MODELS \
    --routes src/lead/routes/eval_sets/degradation_30.txt \
    --conditions none:0 lidar:1.0 camera:1.0 \
    --out "$CSV"
  log "pass $pass_number finished with status $?"
done

log "state after the retries:"
report

log "rewriting the summary"
$PY scripts/common/summarize_closed_loop.py \
  --csv "$CSV" --reference rung2a \
  > "$RES/closed_loop_table.txt" 2>&1
cat "$RES/closed_loop_table.txt"
log "done"
