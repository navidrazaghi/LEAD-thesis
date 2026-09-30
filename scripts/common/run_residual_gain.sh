#!/bin/bash
# The residual-gain rung: one change against the dense control.
#
# WHAT IT TESTS
#
# The thesis decomposed a query's reliance on a modality into two factors --
# the share of attention mass it gives that modality, and the attention's
# authority over what leaves the fusion block. Measured on the gated rung those
# were about 0.50 and 0.40, so the reliance the intervention test found was
# about 0.20. The observability gate could move the first factor and had no
# access to the second, which is the mechanical reason it reallocated attention
# eleven times more than any ungated rung and still lost in closed loop.
#
# This rung targets the other factor. A learned per-token scalar multiplies the
# attention output before it joins the residual stream, so the block can decide
# how much of what it read survives, not only where it read from.
#
# One behaviour follows that the gate cannot provide at all: when both
# modalities are unreliable, reweighting between two bad readings still gives a
# bad reading, while a gain can shut the contribution down and leave the token
# on what it already carried. The thesis lists that case as unaddressed.
#
# WHY IT IS DENSE
#
# The gain needs no modality axis -- it is one scalar per token from the
# normalised stream -- so it works on dense attention exactly as on sparse, and
# the dense operator is what new rungs use. Its control is
# rung2a_dense_curriculum, which is finishing its closed-loop evaluation now,
# and against which this is exactly one change.
#
# A guard in Transfuser.__init__ refuses to build if the flag is set and no gain
# exists, because until this week the flag was read by one backbone and silently
# ignored by every other -- which would have trained something identical to the
# control and blamed the idea.
#
# The lock keeps this off the card until the evaluation has finished with it.
#
# Everything runs inside a function called on the last line: bash reads a script
# incrementally, so editing this file while it runs would corrupt it.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local out="$HOME/LEAD/lead/outputs"
  local final="model_0009.pth"
  local rung=rung6_residual_gain

  # Exactly one change against rung2a_dense_curriculum, which carries the
  # curriculum on the dense operator with use_observability false. Turning the
  # observability head on here as well would have made two changes and the
  # comparison would attribute whatever happened to either of them -- the
  # mistake the control rungs exist to prevent, made while writing the script
  # that depends on them.
  #
  # It also would not have helped: the gain as implemented reads the token
  # stream and nothing else, so it has no use for an observability head yet.
  # Conditioning it on one is the next rung, not this one.
  local common=(
    policy.transfuser.use_residual_gain=true
    training.data.use_sensor_degradation=true
  )

  # rung2d's evaluation is ahead of this in the queue and flock promises
  # exclusion rather than order, so waiting on the lock alone would be a coin
  # toss with it. Wait for its results file to be complete first; that is
  # deterministic. If it is neither running nor started, go ahead.
  local ahead="$HOME/LEAD/lead/results/closed_loop_rung2d.csv"
  echo "[$(date +%H:%M:%S)] waiting for the rung2d evaluation to finish"
  rows_ahead() {
    # The file does not exist until the evaluation writes its first row, and a
    # redirect from a missing file is an error rather than an empty read, so it
    # is checked before it is read.
    [ -f "$ahead" ] || { echo 0; return; }
    echo "$(( $(wc -l < "$ahead") - 1 ))"
  }
  while [ "$(rows_ahead)" -lt 150 ]; do
    ps -eo args --no-headers | grep -qx "bash scripts/common/run_rung2d_eval.sh" || {
      echo "[$(date +%H:%M:%S)] the rung2d evaluation is not running; going ahead"
      break
    }
    sleep 600
  done

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 259200 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  stage() {
    local name=$1; shift
    if [ -f "$out/$name/$final" ]; then
      echo "[$(date +%H:%M:%S)] SKIP $name (already finished)"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] START $name"
    bash scripts/common/run_rung.sh "$name" "$@" || return 1
    [ -f "$out/$name/$final" ] || {
      echo "[$(date +%H:%M:%S)] $name ended without $final"
      return 1
    }
    echo "[$(date +%H:%M:%S)] DONE $name"
  }

  stage "$rung" "${common[@]}" || { echo "pretrain failed"; exit 1; }

  # resume_from_last_checkpoint also decides whether the state-dict load is
  # strict, and it must be false here: the pretrain has no planning decoder, so
  # a strict load would refuse the weights this stage exists to extend.
  stage "${rung}_post" "${common[@]}" \
    policy.transfuser.use_planning_decoder=true \
    training.experiment.resume_from_last_checkpoint=false \
    training.experiment.initial_weights_file="$out/$rung/$final" \
    || { echo "posttrain failed"; exit 1; }

  echo "[$(date +%H:%M:%S)] $rung complete; evaluate ${rung}_post against rung2a_dense_curriculum_post"
}

main "$@"
