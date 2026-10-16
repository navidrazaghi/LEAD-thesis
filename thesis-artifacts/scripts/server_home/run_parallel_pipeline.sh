#!/bin/bash
#
# What runs after the deformable post-train, with evaluation split across CARLA
# instances.
#
# Replaces the tail of run_diverse_deformable.sh (its sequential scoring) and the
# waiting run_diverse_curriculum2.sh. Order:
#
#   1. wait for the deformable post-train process to exit with model_0030.pth
#   2. validate parallel scoring: the reference checkpoint on six routes it
#      scored 100 on sequentially, three CARLA instances at once. Every route
#      must produce a score, and the mean absolute difference from the
#      sequential scores must stay within 20 points. Fail -> one shard, which is
#      the old sequential behaviour.
#   3. score the deformable run, 30 routes x 3 conditions
#   4. train curriculum v2 and score it the same way
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	OPERATOR=$HOME/LEAD/lead/outputs/rung2ad_diverse_post31
	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	ulimit -n 65536

	# --- 1. the post-train still running from the old chain ------------------
	while pgrep -f '[l]ead.training.train.*rung2ad_diverse_post31' > /dev/null; do sleep 120; done
	[ -f "$OPERATOR/model_0030.pth" ] || { say "FATAL: deformable post-train ended without model_0030.pth"; exit 1; }
	say "deformable post-train finished"

	# --- 2. validate parallel scoring ---------------------------------------
	VALIDATION_ROUTES=outputs/eval_shards/validation_routes.txt
	mkdir -p outputs/eval_shards
	$PY - "$VALIDATION_ROUTES" <<'EOF'
import csv, sys
names = open("src/lead/routes/eval_sets/degradation_30.txt").read().split()
hundred = sorted({r["route"] for r in csv.DictReader(open("results/reference_closed_loop.csv"))
                  if r["modality"] == "none" and r["driving_score"].strip()
                  and float(r["driving_score"]) == 100.0 and r["route"] in names})
open(sys.argv[1], "w").write("\n".join(hundred[:6]) + "\n")
print("validation routes:", hundred[:6])
EOF
	rm -f results/parallel_validation.csv
	say "validating: reference on 6 routes, 3 instances"
	$PY ~/eval_parallel.py --models ref0=reference/seed0 --routes "$VALIDATION_ROUTES" \
		--conditions none:0 --out results/parallel_validation.csv --shards 3 --base-port 4000
	SHARDS=1
	if $PY - <<'EOF'
import csv, sys
par = {r["route"]: r["driving_score"] for r in csv.DictReader(open("results/parallel_validation.csv"))}
seq = {r["route"]: float(r["driving_score"]) for r in csv.DictReader(open("results/reference_closed_loop.csv"))
       if r["modality"] == "none" and r["driving_score"].strip()}
routes = open("outputs/eval_shards/validation_routes.txt").read().split()
scored = [r for r in routes if par.get(r, "").strip()]
diffs = [abs(float(par[r]) - seq[r]) for r in scored]
mean = sum(diffs) / len(diffs) if diffs else float("inf")
print(f"parallel validation: {len(scored)}/{len(routes)} scored, mean |diff| {mean:.2f}")
for r in routes:
    print(f"  {r}: parallel {par.get(r, 'missing')} vs sequential {seq.get(r)}")
sys.exit(0 if len(scored) == len(routes) and mean <= 20.0 else 1)
EOF
	then
		SHARDS=3
		say "parallel scoring validated: 3 instances"
	else
		say "parallel scoring NOT validated: falling back to one instance"
	fi

	# --- 3. score the deformable run ----------------------------------------
	say "scoring deformable_diverse, 30 routes x 3 conditions, $SHARDS instance(s)"
	$PY ~/eval_parallel.py --models deformable_diverse="$OPERATOR" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out results/closed_loop_diverse_deformable.csv --shards "$SHARDS" --base-port 5200 \
		|| say "WARNING: deformable scoring reported missing rows; see outputs/eval_shards"
	say "deformable scored: $(( $(wc -l < results/closed_loop_diverse_deformable.csv) - 1 )) rows"

	# --- 4. curriculum v2 ---------------------------------------------------
	SHARDS="$SHARDS" bash ~/run_diverse_curriculum2_parallel.sh
}

main "$@"
