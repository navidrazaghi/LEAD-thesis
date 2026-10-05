#!/bin/bash
# One line describing where rung2d is: pretrain, posttrain, done, or broken.
#
# Every terminal state prints something. A watcher that only reports the happy
# path is silent through a crash, an out-of-memory and a diverged loss, and on a
# run this long silence looks exactly like still training.
#
# The stage is read from which output directory exists rather than from the log,
# because the follow-on script starts the posttrain in a separate process and
# the pretrain's own log says nothing about it.

OUT=~/LEAD/lead/outputs
PRE=$OUT/rung2d_deployment_families
POST=${PRE}_post
FINAL=model_0009.pth

alive() { pgrep -f "[l]ead.training.train" > /dev/null 2>&1; }
armed() { pgrep -f "[r]un_rung2d.sh" > /dev/null 2>&1; }

# Failures first: a crashed stage can leave a plausible progress bar as the last
# thing in its log.
for log in ~/rung2d.log ~/rung2d_chain.log; do
  if grep -qE 'Traceback|CUDA out of memory|RuntimeError|AssertionError|training failed' "$log" 2>/dev/null; then
    echo "FAILED $(grep -E 'Traceback|CUDA out of memory|RuntimeError|AssertionError|training failed' "$log" | tail -1 | cut -c1-150)"
    exit 0
  fi
done

if grep -qiE 'loss[^ ]* *[=:] *(nan|inf)' ~/rung2d.log 2>/dev/null; then
  echo "DIVERGED the loss went non-finite"
  exit 0
fi

if [ -f "$POST/$FINAL" ]; then
  echo "DONE both stages finished; evaluate rung2d_deployment_families_post"
  exit 0
fi

if [ -f "$PRE/$FINAL" ]; then
  if alive; then
    echo "POSTTRAIN pretrain finished, posttrain running $(grep -o 'Epoch [0-9]*: *[0-9]*%' "$(ls -t ~/chain_logs/*post*.log 2>/dev/null | head -1)" 2>/dev/null | tail -1)"
  elif armed; then
    echo "POSTTRAIN_STARTING pretrain finished; the follow-on has not launched the posttrain yet"
  else
    echo "STALLED pretrain finished but nothing is running and no follow-on is armed"
  fi
  exit 0
fi

if alive; then
  echo "PRETRAIN $(grep -o 'step [0-9]* |[^|]*|[^|]*|' ~/rung2d.log 2>/dev/null | tail -1)"
  exit 0
fi

echo "STALLED no training process, and the pretrain never wrote $FINAL"
