#!/bin/bash
#
# After the seed-2 chain: the generalisation-gap measurement, then a subset with
# more logs at the same compute per epoch.
#
# Stage 1 (option 1). Held-out loss for the curriculum models. The gap was
# measured for the two baselines only (9.2x old, 4.7x new); this asks whether
# the degradation curriculum narrows it, which would make it a regulariser and
# not only a robustness lever.
#
# Stage 2 (options 2 + 3). The overfitting lever that is left is scene
# diversity, not parameter count: 312,502 frames from 585 logs are 585 driving
# situations, and consecutive frames of one log are nearly duplicates. So:
#
#   850 logs (24 GB more to download), every scenario type, 12 towns,
#   capped to the same number of scenes per epoch as the 585-log runs
#   (60,672 pretrain, 56,000 post-train) with shuffle_scenes=true.
#
# That holds compute per epoch, the recipe, and the evaluation fixed, and moves
# only how many distinct logs those scenes come from. The model trained is the
# plain dense baseline, so it reads against results/closed_loop_diverse_baseline.csv
# as a one-factor comparison.
#
# Body in a function called on the last line.

main() {
	set -u
	cd ~/LEAD/lead || exit 1

	PY=~/miniconda3/envs/lead/bin/python
	SEL=$HOME/new_subset/selected_850_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung0_diverse850
	POST=$HOME/LEAD/lead/outputs/rung0_diverse850_post31
	CSV=results/closed_loop_diverse850_baseline.csv
	LOG=$HOME/diverse850.log
	SHARDS=${SHARDS:-3}
	SCENES_PRE=60672
	SCENES_POST=56000

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	# --- wait for the seed-2 chain ------------------------------------------
	while pgrep -f '[r]un_diverse_dense_consistency_seed2.sh' > /dev/null; do
		sleep 300
	done
	say "seed-2 chain finished"

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	# --- stage 1: held-out loss of the curriculum models ---------------------
	say "stage 1: held-out loss, 28 logs no run trained on"
	O=$HOME/LEAD/lead/outputs
	for pair in \
		"dense_curriculum2:$O/rung2a_diverse_curriculum2_post31" \
		"deformable_curriculum2:$O/rung2ad_diverse_curriculum2_post31" \
		"consistency_seed0:$O/rung2a_diverse_consistency_post31" \
		"consistency_seed1:$O/rung2a_diverse_consistency_seed1_post31" \
		"consistency_seed2:$O/rung2a_diverse_consistency_seed2_post31"; do
		NAME=${pair%%:*}
		CKPT=${pair#*:}/model_0030.pth
		if [ ! -f "$CKPT" ]; then
			say "  skipping $NAME: no $CKPT"
			continue
		fi
		for LOGS in "$HOME/new_subset/held_out_28.txt" "$HOME/new_subset/train_check_div_28.txt"; do
			say "  $NAME on $(basename "$LOGS" .txt)"
			$PY ~/heldout_loss.py "$NAME" "$CKPT" --logs "$LOGS" >> "$LOG" 2>&1 \
				|| say "  WARNING: $NAME on $(basename "$LOGS") failed; see $LOG"
		done
	done
	say "stage 1 done: results/heldout_loss.csv"

	# --- stage 2: 850-log subset at the same compute -------------------------
	N=$(grep -c . "$SEL")
	[ "$N" -eq 850 ] || { say "FATAL: $N logs in $SEL, expected 850"; exit 1; }

	say "stage 2: downloading the 850-log selection, 4 jobs"
	$PY ~/fetch_selected.py "$SEL" >> "$LOG" 2>&1 \
		|| { say "FATAL: download failed; see $LOG"; exit 1; }

	MISSING=0
	for view in normal_view perturbated_view; do
		for s in $(cat "$SEL"); do
			[ -f "data/lead/123D/logs/$view/$s/sync.arrow" ] || MISSING=$((MISSING + 1))
		done
	done
	[ "$MISSING" -eq 0 ] || { say "FATAL: $MISSING log views without sync.arrow"; exit 1; }
	say "download verified: 850 logs, both views"

	FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
	[ "$FREE" -ge 8 ] || { say "FATAL: only ${FREE}G free after the download"; exit 1; }

	NAMES=$(awk -F/ '{print $2}' "$SEL" | sort)
	LOG_NAMES=$(echo "$NAMES" | paste -sd, -)

	say "building the cache for the new logs"
	$PY -m lead.training.build_cache \
		training.data.force_cache_rebuild=false \
		"training.data.py123d_log_names=[$LOG_NAMES]" \
		>> "$LOG" 2>&1 || { say "FATAL: cache build failed"; tail -20 "$LOG"; exit 1; }
	say "cache built"

	COMMON=(
		training.data.read_from_cache_store=true
		"training.data.py123d_log_names=[$LOG_NAMES]"
		training.data.shuffle_scenes=true
		training.optimization.batch_size=32
		training.lightning.accumulate_grad_batches=2
		training.optimization.num_epochs=31
	)

	if [ ! -f "$PRE/model_0030.pth" ]; then
		say "pretrain starting: 31 epochs, 850 logs, capped at $SCENES_PRE scenes"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.data.max_num_scenes=$SCENES_PRE \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" \
			>> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"

	if [ ! -f "$POST/model_0030.pth" ]; then
		say "post-train starting: 31 epochs, capped at $SCENES_POST scenes"
		$PY -m lead.training.train "${COMMON[@]}" \
			training.data.max_num_scenes=$SCENES_POST \
			policy.transfuser.use_planning_decoder=true \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.initial_weights_file="$PRE/model_0030.pth" \
			training.experiment.output_dir="$POST" \
			>> "$LOG" 2>&1
		[ -f "$POST/model_0030.pth" ] || { say "FATAL: post-train ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "post-train done"

	say "held-out loss of the 850-log baseline"
	$PY ~/heldout_loss.py diverse850 "$POST/model_0030.pth" >> "$LOG" 2>&1 \
		|| say "WARNING: held-out loss failed; see $LOG"
	$PY ~/heldout_loss.py diverse850 "$POST/model_0030.pth" \
		--logs "$HOME/new_subset/train_check_div_28.txt" >> "$LOG" 2>&1 \
		|| say "WARNING: train-check loss failed; see $LOG"

	say "scoring diverse850_baseline, 30 routes x 3 conditions, $SHARDS instance(s)"
	$PY ~/eval_parallel.py --models diverse850_baseline="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out "$CSV" --shards "$SHARDS" --base-port 7600 \
		|| say "WARNING: scoring reported missing rows; see outputs/eval_shards"
	say "done: $(( $(wc -l < "$CSV") - 1 )) rows in $CSV"
	say "read against results/closed_loop_diverse_baseline.csv: same recipe and"
	say "the same scenes per epoch -- only the number of logs they come from differs."
}

main "$@"
