#!/bin/bash
# The baseline, retrained on the recipe LEAD ships rather than the one it got.
#
# WHY
#
# rung0's training loss reaches its floor at epoch two and rises from there,
# ending 2.5x above its best in every tracked head. rung2a_dense_curriculum,
# the only other dense run, does the same and ends 1.4x above its best. Every
# sparse run converges monotonically on the same data and the same epoch budget.
# So the baseline every comparison in the thesis is measured against is a run
# that never converged.
#
# The cause is almost certainly the recipe rather than the architecture, because
# the architecture is LEAD's and LEAD converges with it. Three things were
# changed from what LEAD ships and the learning rate was not one of them:
#
#   batch size   64 -> 8      an eighth of the samples per step, so about 2.8x
#                             the gradient noise
#   epochs       31 -> 10     a third of the optimisation
#   data         all -> 450 logs
#   learning rate 3e-4 -> 3e-4   left at the value tuned for batch 64
#
# A step size chosen for a quiet gradient, applied to one 2.8x noisier, is the
# textbook way to get a loss that finds a floor and then leaves it.
#
# WHAT THIS CHANGES, AND WHAT IT DELIBERATELY DOES NOT
#
# The effective batch is restored to 64 through gradient accumulation: 32 real
# samples per step, accumulated twice. That is arithmetically the batch-64
# gradient, so the shipped learning rate needs no rescaling and no rescaling has
# to be defended. Measured, it peaks at 23.5 GB against 37.3 GB for a true batch
# of 64, which matters because this card is shared -- a 36-hour run that dies at
# hour 30 because a neighbour allocated memory is a run wasted.
#
# The data is deliberately left alone. If the subset were enlarged in the same
# run and the result improved, nothing would say which change did it. The
# evidence also does not point there: seven rungs converge on this same 450-log
# subset.
#
# EPOCHS: 31 FOR THE PRETRAIN, 10 FOR THE POSTTRAIN
#
# Not symmetric, on purpose. The divergence is in the pretrain's loss curve --
# that is where the floor at epoch two and the rise after it happen. The
# posttrain, which adds the planning decoder on top, converged cleanly at ten
# epochs in every rung. Tripling it would cost another 24 hours to fix something
# that is not broken. If the pretrain converges and the driving score is still
# poor, lengthening the posttrain is the next single change.
#
# Throughput is the same at both batch sizes -- 14 samples/s either way -- so
# the bottleneck is the loader or the card, not the batch. That means the cost
# here is set by the epoch count alone: about 70 minutes an epoch, so 36 hours
# for the pretrain and 12 for the posttrain.
#
# The ulimit is not optional. Without it this dies outright at batch 32 with
# "Too many open files" rather than merely crawling, which is how a hand-run
# probe failed before this script existed.

set -u

main() {
  cd "$HOME/LEAD/lead" || exit 1
  ulimit -n 65536

  local out="$HOME/LEAD/lead/outputs"
  local rung=rung0_lead_recipe
  local final_pre="model_0030.pth"
  local final_post="model_0009.pth"

  # Everything LEAD ships, restored. batch_size and accumulate_grad_batches
  # multiply to 64; the learning rate is untouched because of that.
  local common=(
    training.optimization.batch_size=32
    training.lightning.accumulate_grad_batches=2
  )

  echo "[$(date +%H:%M:%S)] waiting for the training lock"
  exec 9>"$HOME/.lead_training.lock"
  flock -w 604800 9 || { echo "timed out waiting for the lock"; exit 200; }
  echo "[$(date +%H:%M:%S)] lock acquired"

  stage() {
    local name=$1 final=$2; shift 2
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

  stage "$rung" "$final_pre" "${common[@]}" \
    training.optimization.num_epochs=31 \
    || { echo "pretrain failed"; exit 1; }

  # resume_from_last_checkpoint also decides whether the state-dict load is
  # strict, and it must be false here: the pretrain has no planning decoder, so
  # a strict load would refuse the weights this stage exists to extend.
  stage "${rung}_post" "$final_post" "${common[@]}" \
    training.optimization.num_epochs=10 \
    policy.transfuser.use_planning_decoder=true \
    training.experiment.resume_from_last_checkpoint=false \
    training.experiment.initial_weights_file="$out/$rung/$final_pre" \
    || { echo "posttrain failed"; exit 1; }

  echo "[$(date +%H:%M:%S)] $rung complete."
  echo "  Read its loss curve before anything else: the question this run exists"
  echo "  to answer is whether the curve descends and flattens, not what it scores."
}

main "$@"
