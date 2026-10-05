#!/bin/bash
# One line describing where the chained evaluation runs currently are.
#
# Printed for a watcher on the laptop, so every terminal state has to produce a
# line: a watcher that only prints good news is silent through a crash, and
# silence looks exactly like still running.

REF=~/reference_eval.log
WEA=~/weather_eval.log

if grep -qE '^\[[0-9]+/[0-9]+\]' "$WEA" 2>/dev/null; then
  echo "WEATHER_STARTED $(grep -cE '^\[[0-9]+/[0-9]+\]' "$WEA") route(s) scored so far: $(tail -1 "$WEA")"
  exit 0
fi

if grep -qE '\] done$' "$REF" 2>/dev/null; then
  echo "REFERENCE_DONE the reference run finished; the weather watcher is settling before it takes the GPU"
  exit 0
fi

if ! pgrep -f run_weather_eval.sh > /dev/null 2>&1; then
  echo "WATCHER_GONE the weather watcher is not running; the chain will not advance on its own"
  exit 0
fi

echo "PENDING $(tail -1 "$REF" 2>/dev/null)"
