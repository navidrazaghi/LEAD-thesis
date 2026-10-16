#!/bin/bash
#
# Held-out loss of the diverse baseline, once its chain has finished.
#
# Two rows, the same pair measured for the old baseline:
#   held_out_28          -- the 28 logs no run trained on
#   train_check_div_28   -- 28 of the diverse run's own training logs, the
#                           control showing the path reproduces its training loss
# Read-only on the checkpoint; appends to results/heldout_loss.csv.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1
	POST=$HOME/LEAD/lead/outputs/rung0_diverse_post31
	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	while pgrep -f '[r]un_diverse_baseline.sh' > /dev/null; do sleep 300; done
	[ -f "$POST/model_0030.pth" ] || { say "FATAL: no diverse post-train checkpoint"; exit 1; }

	# Every 21st selected log: 28 spread across the town-ordered selection.
	awk 'NR % 21 == 1' ~/new_subset/selected_frames_town.txt | head -28 > ~/new_subset/train_check_div_28.txt
	[ "$(wc -l < ~/new_subset/train_check_div_28.txt)" -eq 28 ] || { say "FATAL: control list is not 28 logs"; exit 1; }

	export OMP_NUM_THREADS=1 LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16 LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	ulimit -n 65536
	for list in train_check_div_28 held_out_28; do
		say "held-out loss: diverse on $list"
		~/miniconda3/envs/lead/bin/python ~/heldout_loss.py diverse "$POST/model_0030.pth" \
			--logs ~/new_subset/$list.txt >> ~/heldout_loss.log 2>&1 \
			|| { say "FATAL: $list failed; see ~/heldout_loss.log"; exit 1; }
	done
	say "done:"; grep objective results/heldout_loss.csv
}

main "$@"
