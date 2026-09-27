#!/bin/bash
# Rung 2d: rung 2a's curriculum, widened to two deployment perturbation families.
#
# One change against rung 2a, which is the discipline every rung in this ladder
# follows. rung2a is the control and does not need retraining for it: the
# sampler takes byte-identical draws when no deployment family is requested,
# which is guaranteed by a test rather than by inspection, so the model already
# on disk is the same experiment it always was.
#
# Only occlusion and ego_state are enabled. Latency is implemented and
# deliberately left off here, because it needs future_ego_pose_extra_ticks,
# which lengthens the future window every scene must supply and so drops the
# scenes near the end of a log. That changes the training set, and a rung whose
# data differs from its control measures the data as much as the augmentation.
# It earns a rung of its own once this one has an answer.
#
# What this is meant to show: the degradation curriculum was the biggest win in
# the thesis, and it only ever damaged how a sensor renders the world. A
# deployed stack also fails with the rendering intact -- a blocked lens, a
# drifting fix, a slow speedometer. Whether covering those helps is the
# question; the conditions are ones the model still drives in, which is what the
# caution governor's regime turned out not to be.
#
# A rung is two stages, not one. The pretrain learns perception; the posttrain
# adds the planning decoder and is what evaluation loads. Running only the first
# leaves no _post checkpoint and nothing to score, which is how this script was
# first written and launched.
#
# Both stages skip themselves if their final checkpoint already exists, so this
# is safe to re-run after an interruption and safe to run while the pretrain
# from an earlier launch is still going.
#
# The ulimit is not optional: the default 1024 starves the LMDB cache readers
# and training crawls at a fiftieth of the speed without ever failing outright.

set -u
cd "$HOME/LEAD/lead" || exit 1
ulimit -n 65536

OUT="$HOME/LEAD/lead/outputs"
FINAL_CHECKPOINT="model_0009.pth"
RUNG=rung2d_deployment_families
DEFORMABLE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

COMMON=(
  policy.transfuser.backbone_target="$DEFORMABLE"
  policy.transfuser.deformable_calibrated_reference=true
  training.data.use_sensor_degradation=true
  training.data.deployment_perturbation_families="[occlusion,ego_state]"
)

# Wait out any training already on the GPU, including this rung's own pretrain
# if it was launched separately.
while pgrep -f "[l]ead.training.train" > /dev/null 2>&1; do
  sleep 300
done

stage() {
  local name=$1; shift
  if [ -f "$OUT/$name/$FINAL_CHECKPOINT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $name (already finished)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START $name"
  bash scripts/common/run_rung.sh "$name" "$@" || return 1
  [ -f "$OUT/$name/$FINAL_CHECKPOINT" ] || {
    echo "[$(date +%H:%M:%S)] $name ended without $FINAL_CHECKPOINT"
    return 1
  }
  echo "[$(date +%H:%M:%S)] DONE $name"
}

stage "$RUNG" "${COMMON[@]}" || { echo "pretrain failed"; exit 1; }

# resume_from_last_checkpoint also decides whether the state-dict load is
# strict, and it must be false here: the pretrain has no planning decoder, so a
# strict load would refuse the weights this stage exists to extend.
stage "${RUNG}_post" "${COMMON[@]}" \
  policy.transfuser.use_planning_decoder=true \
  training.experiment.resume_from_last_checkpoint=false \
  training.experiment.initial_weights_file="$OUT/$RUNG/$FINAL_CHECKPOINT" \
  || { echo "posttrain failed"; exit 1; }

echo "[$(date +%H:%M:%S)] rung2d complete; evaluate ${RUNG}_post"
