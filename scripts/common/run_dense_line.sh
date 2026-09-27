#!/bin/bash
# The dense line: every rung after the deformable operator was dropped.
#
# WHY THIS EXISTS
#
# The deformable operator was adopted for cost and did the opposite. Measured on
# an A100 it is 13% slower end to end (results/cost.csv), and the profile says
# why: dense fusion attention is 2.2% of the forward pass and the deformable
# operator turns that into 36.6% (results/forward_profile.txt). At this token
# count there was nothing to reclaim, so new work uses the dense operator.
#
# That decision costs one training run, and this script is it.
#
# THE CONTROL PROBLEM
#
# Every rung of the thesis ladder -- 1, 2, 2a, 2b, 3, 4 -- is deformable. The
# only dense rung is rung0, which has no degradation curriculum. So a new dense
# rung has nothing to be compared against: scoring it against rung2a would
# change the operator and the augmentation in one step, which is exactly what
# the ladder's one-change-per-rung discipline exists to prevent.
#
# rung2a_dense_curriculum is that missing control -- dense plus the curriculum,
# nothing else. It is not pure overhead. It answers a question the thesis cannot
# currently answer: whether the curriculum's win, which is the thesis's largest
# positive result, ever depended on the operator at all. Against rung2a it
# isolates the operator under the curriculum; against rung0 it re-measures the
# curriculum on the dense path.
#
# WHAT DENSE CAN AND CANNOT CARRY
#
# The observability head works: it hangs off the top-down pyramid, not off the
# fusion operator, and transfuser_backbone.py builds that pyramid whenever
# use_observability is set. The gate does not, and does not fail quietly --
# transfuser.py raises, because the gate biases a softmax over a modality axis
# and the dense operator has no such axis. The gate was the thesis's negative
# result, so nothing is lost, but no rung on this line can use it.
#
# backbone_target is deliberately not passed. Dense is the config default, and
# naming it here would invite someone to keep the deformable knobs beside it;
# those knobs are read only by the deformable backbone and are silently ignored
# everywhere else, so a stray deformable_calibrated_reference=true on this line
# would look meaningful and do nothing.
#
# A rung is two stages. The pretrain learns perception; the posttrain adds the
# planning decoder and is what evaluation loads. Running only the first leaves
# no _post checkpoint and nothing to score.
#
# Both stages skip themselves if their final checkpoint exists, so this is safe
# to re-run after an interruption.
#
# The ulimit is not optional: the default 1024 starves the LMDB cache readers
# and training crawls at a fiftieth of the speed without ever failing outright.

set -u

OUT="$HOME/LEAD/lead/outputs"
FINAL_CHECKPOINT="model_0009.pth"

# WAITING FOR THE GPU IS HARDER THAN IT LOOKS
#
# This waited by polling for trainer processes and starting when none was found.
# That is a race: a rung is two stages, and between them there are seconds where
# the pretrain has exited and the posttrain has not begun. A poll landing there
# sees an idle card and launches onto one about to be busy again. Requiring
# several consecutive clear polls narrowed the window without closing it, and it
# still had to name every driver script it might be waiting for -- which meant a
# driver under a new name was invisible to it.
#
# A lock has neither problem. Whoever trains holds it; whoever waits blocks on
# it; nothing has to recognise anything by name. The poll is gone.
#
# There is one thing a lock cannot do, and it is worth being honest about: it
# only serialises against processes that take it. Anything launched by hand
# without the lock is still invisible, which is why every driver in this
# directory takes it.
#
# EDITING A RUNNING SCRIPT
#
# Everything below runs inside a function. Bash reads a script incrementally,
# remembering a byte offset, so editing a file while it runs makes the offset
# point into the wrong line and the next command it reads is garbage. That is
# how rung2d lost its posttrain: the file was corrected thirteen minutes after
# it started, and fourteen hours later bash came back to the file and failed on
# a line that was valid. Bash must parse a function through to its closing brace
# before executing any of it, so the whole file is in memory before anything
# runs, and a later edit cannot reach it.

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  exec 9>"$HOME/.lead_training.lock"
  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  flock -w 172800 9 || { echo "timed out waiting for the training lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

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

  rung() {
    local name=$1; shift
    stage "$name" "$@" || return 1
    # resume_from_last_checkpoint also decides whether the state-dict load is
    # strict, and it must be false here: the pretrain has no planning decoder, so
    # a strict load would refuse the weights this stage exists to extend.
    stage "${name}_post" "$@" \
      policy.transfuser.use_planning_decoder=true \
      training.experiment.resume_from_last_checkpoint=false \
      training.experiment.initial_weights_file="$OUT/$name/$FINAL_CHECKPOINT" \
      || return 1
  }

  # The control. One change against rung0: the curriculum.
  rung rung2a_dense_curriculum \
    training.data.use_sensor_degradation=true \
    || { echo "dense control failed"; exit 1; }

  # Later dense rungs go here, one change each against the control above. The
  # obvious next one is the deployment families, which is rung2d's question asked
  # on this operator:
  #
  #   rung rung2d_dense_deployment \
  #     training.data.use_sensor_degradation=true \
  #     training.data.deployment_perturbation_families="[occlusion,ego_state]"
  #
  # It is left commented rather than queued, because rung2d is answering that
  # question on the deformable line first and its answer decides whether asking it
  # again here is worth thirty-five hours.

  echo "[$(date +%H:%M:%S)] dense line complete; evaluate rung2a_dense_curriculum_post"
}

main "$@"
