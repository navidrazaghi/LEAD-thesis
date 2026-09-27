#!/bin/bash
# Rung 2d's posttrain, which the original driver never reached.
#
# WHY THIS FILE EXISTS RATHER THAN A RE-RUN OF run_rung2d.sh
#
# The pretrain finished and wrote model_0009.pth. The posttrain never started,
# because the driver died at the handover with a syntax error on a line that is
# and always was valid:
#
#   run_rung2d.sh: line 61: syntax error near unexpected token `('
#
# Bash does not load a script into memory. It reads it as it goes, remembering a
# byte offset, and re-reads from that offset when it needs the next command.
# run_rung2d.sh was edited thirteen minutes after that driver started -- the edit
# that added the posttrain stage in the first place -- so fourteen hours later,
# when the pretrain returned and bash went back to the file for the next
# command, the offset it had remembered pointed into the middle of a different
# line than the one it was written against.
#
# A relaunch of the same script would work. This one exists so the two stages
# are not entangled again, and so the guard below is written down somewhere.
#
# THE GUARD
#
# Everything runs inside a function. Bash has to parse a function definition
# through to its closing brace before it can execute any of it, so the whole
# file is read and turned into a parse tree up front, and editing the file
# afterwards cannot reach the running program. The call is the last line.
#
# The lock is the other half: it serialises this against the dense line, which
# waits on the same file rather than polling for processes.
#
# The ulimit is not optional: the default 1024 starves the LMDB cache readers
# and training crawls at a fiftieth of the speed without ever failing outright.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local out="$HOME/LEAD/lead/outputs"
  local final="model_0009.pth"
  local rung="rung2d_deployment_families"
  local deformable=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

  if [ ! -f "$out/$rung/$final" ]; then
    echo "[$(date +%H:%M:%S)] no pretrain checkpoint at $out/$rung/$final; nothing to extend"
    exit 1
  fi
  if [ -f "$out/${rung}_post/$final" ]; then
    echo "[$(date +%H:%M:%S)] SKIP ${rung}_post (already finished)"
    exit 0
  fi

  # Blocks until whatever else is training lets go. 200 is this script's own
  # code for "someone else holds it", distinct from any training failure.
  exec 9>"$HOME/.lead_training.lock"
  flock -w 172800 9 || { echo "timed out waiting for the training lock"; exit 200; }

  echo "[$(date +%H:%M:%S)] START ${rung}_post"
  # resume_from_last_checkpoint also decides whether the state-dict load is
  # strict, and it must be false here: the pretrain has no planning decoder, so
  # a strict load would refuse the weights this stage exists to extend.
  bash scripts/common/run_rung.sh "${rung}_post" \
    policy.transfuser.backbone_target="$deformable" \
    policy.transfuser.deformable_calibrated_reference=true \
    training.data.use_sensor_degradation=true \
    training.data.deployment_perturbation_families="[occlusion,ego_state]" \
    policy.transfuser.use_planning_decoder=true \
    training.experiment.resume_from_last_checkpoint=false \
    training.experiment.initial_weights_file="$out/$rung/$final" \
    || { echo "[$(date +%H:%M:%S)] posttrain failed"; exit 1; }

  [ -f "$out/${rung}_post/$final" ] || {
    echo "[$(date +%H:%M:%S)] ${rung}_post ended without $final"
    exit 1
  }
  echo "[$(date +%H:%M:%S)] DONE ${rung}_post; evaluate it"
}

main "$@"
