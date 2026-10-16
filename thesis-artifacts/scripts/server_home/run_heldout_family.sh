#!/bin/bash
#
# Score the ladder under a degradation family no checkpoint ever trained on.
#
# apply_sensor_degradation() takes a deployment_families argument and nothing
# in the training path ever passes it, so every one of the 23 runs trained on
# the appearance families alone: camera gain, blur and noise, and lidar point
# dropout. The two deployment families -- patch occlusion and ego-state fault
# -- are implemented, reachable from evaluation through degrade_batch_family(),
# and have never been seen by any model here.
#
# That makes them a held-out test set for the curriculum, which is the one
# question rung 2a cannot currently answer: whether it bought robustness or
# memorised the corruption it was trained on. Both outcomes are worth having.
# If 2a still leads under occlusion the curriculum generalises; if the two
# rungs converge, the gain is family-specific and the thesis says so first.
#
# Only rungs 0 and 2a can take part. The observability gate and the
# hallucination head both raise on a non-none degrade_family by design -- a
# spatial fault has no scalar severity to put on the modality axis -- so rungs
# 4 and 2c are out of this comparison. Rung 2a is the one that mattered anyway.
#
# Cost is 30 routes x 2 models x 2 conditions. Both conditions are scored here
# rather than reading the intact column off results/closed_loop.csv, because
# that column was collected on a different day under different machine load,
# and comparing across sweeps is the error that produced the contention factor
# we had to retract. To halve the run at the cost of that guarantee, drop
# none:0 from CONDITIONS.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift;
# parsing the whole file first makes that impossible.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	CSV=results/closed_loop_heldout_family.csv
	FAMILY=${1:-occlusion}
	CONDITIONS=("none:0" "$FAMILY:1.0")

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	export LEAD_RUNTIME_TYPE_CHECKING=false
	export TIMM_USE_OLD_CACHE=1
	export WANDB_MODE=offline
	export OMP_NUM_THREADS=1
	ulimit -n 65536

	# run_evaluation.py routes a condition name to degrade_family instead of
	# degrade_modality only if it is in its _DEPLOYMENT_FAMILIES tuple. A name
	# outside it is silently treated as a modality, which would score the wrong
	# thing and look like a result, so refuse anything the harness will not
	# recognise as a family.
	if [ "$FAMILY" != "occlusion" ] && [ "$FAMILY" != "ego_state" ]; then
		say "FATAL: '$FAMILY' is not a deployment family."
		say "  Use occlusion or ego_state; anything else is read as a modality."
		exit 1
	fi

	# Both checkpoints must exist before a sweep that costs hours starts, and
	# both must be the post-trained ones the ladder was scored on.
	for dir in outputs/rung0_baseline_post outputs/rung2a_curriculum_only_post; do
		if [ -z "$(ls "$dir"/model_*.pth 2>/dev/null)" ]; then
			say "FATAL: no checkpoint in $dir"
			exit 1
		fi
	done

	say "held-out family sweep: $FAMILY"
	say "  no rung trained on this family; deployment_families was never set"
	say "  rungs 4 and 2c excluded: gate and hallucination head raise on it"
	say "  conditions: ${CONDITIONS[*]}"

	# --out keeps existing rows and skips them, so this is resumable: if the
	# server goes away mid-sweep, rerunning picks up where it stopped.
	$PY scripts/common/run_evaluation.py \
		--models rung0=outputs/rung0_baseline_post \
		         rung2a=outputs/rung2a_curriculum_only_post \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions "${CONDITIONS[@]}" \
		--out "$CSV" \
		>> ~/heldout_family_eval.log 2>&1

	if [ ! -f "$CSV" ]; then
		say "FATAL: no results written; see ~/heldout_family_eval.log"
		exit 1
	fi

	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read the $FAMILY column as rung2a minus rung0, against the same"
	say "difference in the none column collected in this same sweep."
	say "NOTE: 30 routes gives an MDE of 17-23 DS. A null here means no large"
	say "  effect, not no effect, and must be written that way."
}

main "$@"
