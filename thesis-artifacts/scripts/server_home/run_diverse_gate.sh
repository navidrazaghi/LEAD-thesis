#!/bin/bash
#
# The observability gate on the diverse subset, and the oracle that tests it.
#
# The thesis's gate rung was trained under the ladder's early recipe (batch 8,
# 10 epochs, curriculum v1) on the old 450 logs, and its closed-loop cost under
# camera destruction could be the gate, or the recipe, or the data. This trains
# it the way every diverse-campaign model was trained, as exactly model 3 of
# that campaign -- deformable fusion, calibrated reference, curriculum v2, the
# published 31 + 31 recipe on the 585 logs -- with the observability head and
# the gate switched on and the gate target left at "logit", the thesis's own.
# Model 3's closed-loop results are therefore its no-gate control.
#
# Then the same checkpoint is driven twice: with the gate it learned, and with
# the oracle gate (evaluation.inference.oracle_gate), which replaces the gate's
# estimate with the reliability the harness actually applied. Both runs share
# one set of weights, so any difference between them is the estimator alone,
# with no training noise in it. If the oracle does not beat the learned gate,
# the estimator was not what failed; if it does not beat the no-gate control
# either, reweighting by observability does not help this policy drive.
#
# The observability targets must be in the cache store, and the store the other
# runs read was built without them. This builds a separate store for this run
# rather than touching that one.
#
# Waits for stage 0: the GPU is needed whole. Smoke-trains before the real run,
# so a configuration error costs minutes rather than a day.
#
# Body in a function called on the last line.

main() {
	set -u
	B=/home/new_drive/razaghi
	cd ~/LEAD/lead || exit 1
	PY=~/miniconda3/envs/lead/bin/python
	SEL=thesis-artifacts/provenance/new_subset/selected_frames_town.txt
	PRE=$HOME/LEAD/lead/outputs/rung3ad_diverse_gate
	POST=$HOME/LEAD/lead/outputs/rung3ad_diverse_gate_post31
	SMOKE=$HOME/LEAD/lead/outputs/smoke_gate
	LOG=$B/diverse_gate.log

	say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

	say "waiting for stage 0"
	while pgrep -f '^bash stage0_run_v2.sh' > /dev/null || pgrep -f '[l]ead.training.train' > /dev/null \
		|| pgrep -f '[e]val_parallel' > /dev/null; do
		sleep 120
	done

	export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
	export NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue
	export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	export LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 WANDB_MODE=offline
	export LIBRARY_PATH="$HOME/.local/cuda-stubs:${LIBRARY_PATH:-}"
	export LD_LIBRARY_PATH="$B/lib/turbojpeg3:${LD_LIBRARY_PATH:-}"
	export SLURM_JOB_ID=1 SLURM_CPUS_PER_TASK=16
	ulimit -n 65536

	NAMES=$(awk -F/ '{print $2}' "$SEL" | sort)
	[ "$(echo "$NAMES" | wc -l)" -eq 585 ] || { say "FATAL: selection is not 585 logs"; exit 1; }
	LOG_NAMES=$(echo "$NAMES" | paste -sd, -)

	# Model 3 of the diverse campaign, argument for argument
	# (run_diverse_curriculum2_parallel.sh) ...
	COMMON=(
		policy.transfuser.backbone_target=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone
		policy.transfuser.deformable_calibrated_reference=true
		policy.transfuser.deformable_learn_cross_reference=true
		policy.transfuser.deformable_num_points=4
		training.data.use_sensor_degradation=true
		training.data.sensor_degradation_probability=0.30
		training.data.sensor_degradation_independent_modalities=true
		training.data.sensor_degradation_full_failure_probability=0.25
		training.data.sensor_degradation_misalignment_probability=0.10
		training.data.read_from_cache_store=true
		"training.data.py123d_log_names=[$LOG_NAMES]"
		training.optimization.batch_size=32
		training.lightning.accumulate_grad_batches=2
		training.optimization.num_epochs=31
	)
	# ... plus the observability head and the gate, as the thesis's gate rung
	# had them (configs/rung3_observability_gated*.yaml), reading a store that
	# carries their targets.
	GATE=(
		policy.transfuser.use_observability=true
		policy.transfuser.observability_soft_targets=true
		policy.transfuser.observability_head_channels=64
		policy.transfuser.use_observability_gate=true
		policy.transfuser.observability_gate_loss_weight=1.0
		policy.transfuser.observability_gate_target=logit
		policy.transfuser.cache_store_dir_name=transfuser_training_cache_obs
	)

	# --- 1. the store with observability targets ---------------------------
	say "building the observability cache store (separate from the shared one)"
	$PY -m lead.training.build_cache training.data.force_cache_rebuild=false "${COMMON[@]}" "${GATE[@]}" \
		>> "$LOG" 2>&1 || { say "FATAL: cache build failed"; tail -20 "$LOG"; exit 1; }
	say "observability store built"

	# --- 2. smoke: both stages, a few batches each --------------------------
	rm -rf "$SMOKE" "${SMOKE}_post"
	$PY -m lead.training.train "${COMMON[@]}" "${GATE[@]}" training.optimization.num_epochs=1 \
		training.lightning.limit_train_batches=20 training.experiment.output_dir="$SMOKE" >> "$LOG" 2>&1
	ls "$SMOKE"/model_*.pth > /dev/null 2>&1 || { say "FATAL: gate smoke pretrain wrote no checkpoint"; tail -30 "$LOG"; exit 1; }
	$PY -m lead.training.train "${COMMON[@]}" "${GATE[@]}" policy.transfuser.use_planning_decoder=true \
		training.optimization.num_epochs=1 training.lightning.limit_train_batches=20 \
		training.experiment.initial_weights_file="$(ls "$SMOKE"/model_*.pth | tail -1)" \
		training.experiment.output_dir="${SMOKE}_post" >> "$LOG" 2>&1
	ls "${SMOKE}_post"/model_*.pth > /dev/null 2>&1 || { say "FATAL: gate smoke post-train wrote no checkpoint"; tail -30 "$LOG"; exit 1; }
	say "smoke passed in both stages"

	# --- 3. the real run --------------------------------------------------------
	if [ ! -f "$PRE/model_0030.pth" ]; then
		say "pretrain starting: 31 epochs, model 3 + observability gate"
		$PY -m lead.training.train "${COMMON[@]}" "${GATE[@]}" \
			training.experiment.resume_from_last_checkpoint=true \
			training.experiment.output_dir="$PRE" >> "$LOG" 2>&1
		[ -f "$PRE/model_0030.pth" ] || { say "FATAL: pretrain ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "pretrain done"
	if [ ! -f "$POST/model_0030.pth" ]; then
		say "post-train starting: 31 epochs from the pretrain"
		$PY -m lead.training.train "${COMMON[@]}" "${GATE[@]}" \
			policy.transfuser.use_planning_decoder=true \
			training.experiment.resume_from_last_checkpoint=false \
			training.experiment.initial_weights_file="$PRE/model_0030.pth" \
			training.experiment.output_dir="$POST" >> "$LOG" 2>&1
		[ -f "$POST/model_0030.pth" ] || { say "FATAL: post-train ended without model_0030.pth"; tail -20 "$LOG"; exit 1; }
	fi
	say "post-train done"

	# --- 4. learned gate, then the oracle, on one checkpoint ----------------
	say "scoring the learned gate: 30 routes x 3 conditions"
	$PY $B/eval_parallel_v3.py --models diverse_gate="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions none:0 lidar:1.0 camera:1.0 \
		--out results/closed_loop_diverse_gate.csv --shards 4 \
		|| say "WARNING: learned-gate scoring reported missing rows"
	# Intact is not repeated: with nothing damaged the oracle hands the gate no
	# signal, so the run would be the learned one again.
	say "scoring the oracle gate: 30 routes x 2 damaged conditions"
	$PY $B/eval_parallel_v3.py --models diverse_gate_oracle="$POST" \
		--routes src/lead/routes/eval_sets/degradation_30.txt \
		--conditions lidar:1.0 camera:1.0 \
		--config evaluation.inference.oracle_gate=true \
		--out results/closed_loop_diverse_gate_oracle.csv --shards 4 \
		|| say "WARNING: oracle scoring reported missing rows"
	$PY $B/gate_summary.py
	say "gate experiment finished"
}

main "$@"
