#!/bin/bash
#
# Score the diverse baseline and the reference checkpoint on scenario_44 --
# one route per Bench2Drive scenario type, intact sensors -- once the diverse
# chain (run_diverse_baseline.sh) has finished. A separate script because the
# chain is already running and must not be edited under bash's feet.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	POST=$HOME/LEAD/lead/outputs/rung0_diverse_post31
	CSV=results/closed_loop_scenario44.csv
	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	while pgrep -f '[r]un_diverse_baseline.sh' > /dev/null; do sleep 300; done
	if [ ! -f "$POST/model_0030.pth" ] || [ ! -f results/closed_loop_diverse_baseline.csv ]; then
		say "FATAL: the diverse chain did not finish; see ~/diverse_chain.log"
		exit 1
	fi
	say "scoring diverse + reference on scenario_44, intact"
	$PY scripts/common/run_evaluation.py \
		--models diverse="$POST" ref0=reference/seed0 \
		--routes src/lead/routes/eval_sets/scenario_44.txt \
		--conditions none:0 \
		--out "$CSV" \
		>> ~/scenario44_eval.log 2>&1
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
}

main "$@"
