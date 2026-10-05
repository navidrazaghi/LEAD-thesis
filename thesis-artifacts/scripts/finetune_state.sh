#!/bin/bash
# One line describing where the ensemble fine-tune is.
#
# Every terminal state prints something. A watcher that only reports success is
# silent through a crash, an out-of-memory and a diverged loss, and silence
# looks exactly like still training -- which on a multi-hour run means finding
# out the next morning.
#
# Completion is "the final epoch's model file exists", not "N model files
# exist". The trainer prunes each epoch's weights once the next epoch is on
# disk, keeping only the epochs listed in epochs_to_keep_checkpoints_for, which
# defaults to empty. So a successful three-epoch run leaves exactly one model
# file behind, and counting them would have reported a finished run as a failed
# one.
#
# The epoch count comes from the run's own config rather than from this script,
# so a fine-tune launched with a different number of epochs is still read
# correctly.

LOG=~/ensemble_finetune.log
OUT=~/LEAD/lead/outputs/rung4_light_auxiliary_post_ensemble

alive() { pgrep -f "[l]ead.training.train" > /dev/null 2>&1; }

# Failures first: a crashed run can still leave a plausible-looking progress bar
# as the last thing in the log.
if grep -qE 'Traceback|CUDA out of memory|RuntimeError|AssertionError' "$LOG" 2>/dev/null; then
  echo "FAILED $(grep -E 'Traceback|CUDA out of memory|RuntimeError|AssertionError' "$LOG" | tail -1 | cut -c1-160)"
  exit 0
fi

# A diverged loss is not an error the trainer raises; it just stops meaning
# anything, so it has to be looked for directly.
if grep -qiE 'loss[^ ]* *[=:] *(nan|inf)' "$LOG" 2>/dev/null; then
  echo "DIVERGED the loss went non-finite; the checkpoint is not worth keeping"
  exit 0
fi

EPOCHS=$(awk '/^ *num_epochs:/ {print $2; exit}' "$OUT/config.yaml" 2>/dev/null)
if [ -z "$EPOCHS" ]; then
  # Before the config is written there is nothing to compare against, so say so
  # rather than guessing an epoch count and reporting against it.
  if alive; then
    echo "PENDING the run has not written its config yet"
  else
    echo "STOPPED the process is gone and no run config was ever written"
  fi
  exit 0
fi

FINAL=$(printf 'model_%04d.pth' "$((EPOCHS - 1))")
PRESENT=$(ls "$OUT"/model_*.pth 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')

if ! alive; then
  if [ -f "$OUT/$FINAL" ]; then
    echo "DONE all $EPOCHS epochs written; final checkpoint $FINAL"
  else
    echo "STOPPED the process is gone without $FINAL; on disk: ${PRESENT:-nothing}"
  fi
  exit 0
fi

PROGRESS=$(grep -o 'Epoch [0-9]*: *[0-9]*%[^]]*]' "$LOG" 2>/dev/null | tail -1)
# The epoch number is the second field on purpose: the watcher reads it to
# decide whether this is a new epoch or the same one it already reported.
EPOCH_NUM=$(grep -o 'Epoch [0-9]*:' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]\+')
echo "EPOCH ${EPOCH_NUM:-0} of $EPOCHS | ${PROGRESS:-starting} | on disk: ${PRESENT:-nothing}"
