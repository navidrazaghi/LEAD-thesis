#!/bin/bash
# Run the ablation ladder to completion, one rung at a time.
#
# Each rung is two runs: a pretrain (perception only) and a posttrain that
# starts from the pretrain's weights and adds the planning decoder. A rung
# cannot start its posttrain until its own pretrain has finished, so the whole
# thing is serial by construction.
#
# Re-runnable. A stage whose final checkpoint already exists is skipped, so an
# interrupted chain picks up where it stopped rather than redoing hours of work.
# A stage that fails stops the chain: a posttrain launched after a failed
# pretrain would either die on the missing weights file or, worse, quietly
# train from scratch and look like a valid rung.
#
#   nohup setsid scripts/common/run_ablation_chain.sh > ~/chain.log 2>&1 &
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/outputs"
LOGS="$HOME/chain_logs"
# 10 epochs, zero-indexed: this is what a finished stage leaves behind.
FINAL_CHECKPOINT="model_0009.pth"

DEFORMABLE="lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone"

mkdir -p "$LOGS"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Wait for any training already running, so the chain can be started while the
# current rung is still going.
wait_for_idle() {
  local pid
  while true; do
    pid=$(ls -l /proc/*/exe 2>/dev/null | grep "envs/lead/bin/python" \
          | head -1 | sed "s|.*/proc/\([0-9]*\)/exe.*|\1|")
    [ -z "$pid" ] && break
    log "waiting on training pid $pid"
    while [ -d "/proc/$pid" ]; do sleep 60; done
  done
}

# run_stage <name> [overrides...]
run_stage() {
  local name=$1; shift
  if [ -f "$OUT/$name/$FINAL_CHECKPOINT" ]; then
    log "SKIP $name (already has $FINAL_CHECKPOINT)"
    return 0
  fi
  log "START $name"
  if "$HERE/run_rung.sh" "$name" "$@" > "$LOGS/$name.log" 2>&1; then
    if [ -f "$OUT/$name/$FINAL_CHECKPOINT" ]; then
      log "DONE  $name"
      return 0
    fi
    log "FAIL  $name (exited cleanly but wrote no $FINAL_CHECKPOINT)"
    return 1
  fi
  log "FAIL  $name (see $LOGS/$name.log)"
  return 1
}

# run_rung <rung-name> [pretrain overrides...]
# The posttrain inherits the same overrides and adds the planning decoder.
# resume_from_last_checkpoint must be false there: it doubles as the strict flag
# for load_state_dict, and a pretrain checkpoint has no planning decoder in it.
run_rung() {
  local rung=$1; shift
  run_stage "${rung}" "$@" || return 1
  run_stage "${rung}_post" "$@" \
    policy.transfuser.use_planning_decoder=true \
    training.experiment.resume_from_last_checkpoint=false \
    training.experiment.initial_weights_file="$OUT/${rung}/$FINAL_CHECKPOINT" || return 1
}

log "chain starting"
wait_for_idle

# Rung 0's pretrain is already done; its posttrain may still be, so it is
# listed here and skipped if it finished.
run_stage rung0_baseline_post \
  policy.transfuser.use_planning_decoder=true \
  training.experiment.resume_from_last_checkpoint=false \
  training.experiment.initial_weights_file="$OUT/rung0_baseline/$FINAL_CHECKPOINT" \
  || { log "chain stopped at rung0 posttrain"; exit 1; }

# Rung 1: sparse fusion, but the cross-modal reference is learned from scratch.
# Isolates what sparsity alone costs or buys.
run_rung rung1_deformable_free \
  policy.transfuser.backbone_target="$DEFORMABLE" \
  || { log "chain stopped at rung1"; exit 1; }

# Rung 2: the same operator, with the reference seeded from the rig geometry.
# Against rung 1, this is what the calibration prior is worth.
run_rung rung2_deformable_calibrated \
  policy.transfuser.backbone_target="$DEFORMABLE" \
  policy.transfuser.deformable_calibrated_reference=true \
  || { log "chain stopped at rung2"; exit 1; }

# Rung 3: the full method. The observability head, the gate that reads it, and
# the degradation curriculum that gives the gate something to learn from. These
# three move together on purpose — the curriculum without the gate is just
# augmentation, and the gate without it has no signal.
run_rung rung3_observability_gated \
  policy.transfuser.backbone_target="$DEFORMABLE" \
  policy.transfuser.deformable_calibrated_reference=true \
  policy.transfuser.use_observability=true \
  policy.transfuser.use_observability_gate=true \
  training.data.use_sensor_degradation=true \
  || { log "chain stopped at rung3"; exit 1; }

log "chain finished: all rungs complete"
