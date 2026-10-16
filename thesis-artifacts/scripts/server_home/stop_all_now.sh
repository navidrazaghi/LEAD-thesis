#!/bin/bash
# Manual stop of everything of ours, requested by the user, so the NVIDIA
# kernel module can be reloaded. Patterns use [.] so they cannot match the
# command line of whatever launched this script.
LOG=$HOME/stop_all_now.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
kill_tree() { local c; for c in $(ps -o pid= --ppid "$1"); do kill_tree "$c"; done; kill -KILL "$1" 2>/dev/null; }
say 'manual stop requested by the user'
for p in $(pgrep -u "$USER" -f 'stop_at_boundary(_v2)?[.]sh'); do say "  boundary watcher $p: TERM"; kill -TERM "$p"; done
for pat in 'run_loader_benchmark[.]sh' 'run_final_chain3[.]sh' 'run_mask_recipe[.]sh' 'run_evaluation[.]py'; do
  pids=$(pgrep -u "$USER" -f "$pat"); if [ -n "$pids" ]; then say "  $pat: TERM "$pids; kill -TERM $pids 2>/dev/null; fi
done
sleep 2
for p in $(pgrep -u "$USER" -f 'training/train[.]py'); do say "  trainer tree rooted at $p: KILL"; kill_tree "$p"; done
sleep 3
for p in $(pgrep -u "$USER" -f 'training/train[.]py'); do say "  leftover trainer $p: KILL"; kill_tree "$p"; done
if ps -o args= -p 2724251 2>/dev/null | grep -q tecton; then say '  ten-day-old self-matching wait loop 2724251: TERM'; kill -TERM 2724251; fi
sleep 3
left=$(pgrep -u "$USER" -f 'training/train[.]py|run_mask_recipe[.]sh|run_final_chain3[.]sh|run_loader_benchmark[.]sh|run_evaluation[.]py|stop_at_boundary' | wc -l)
say "our queue processes left: $left"
held=''; for p in $(pgrep -u "$USER"); do ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/nvidia && held="$held $p"; done
say "our processes holding /dev/nvidia*: ${held:-none}"
for p in $held; do ps -o pid=,args= -p $p | cut -c1-120 | tee -a "$LOG"; done
say "omati python processes still present: $(pgrep -u omati -f python | wc -l)"
ps -u omati -o pid=,etime=,args= | grep -i python | cut -c1-110 | tee -a "$LOG"
say 'checkpoints left behind by the mask run:'
ls -l --time-style=+'%H:%M' $HOME/LEAD/lead/outputs/mask_lead_recipe/ $HOME/LEAD/lead/outputs/mask_lead_recipe_post/ 2>&1 | grep -vE '^total|wandb|config' | tee -a "$LOG"
say 'STOPPED'
