#!/bin/bash
# v2 of the boundary stop. Same trigger as v1 -- the driver's own 'stage 1 done'
# line -- but each trainer's whole process tree is killed (loader workers,
# torch.compile's worker pool, the wandb helpers), not only what matches the
# trainer's command line, so nothing of ours is left holding /dev/nvidia when
# the kernel module is reloaded.
LOG=$HOME/boundary_stop.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say 'v2 armed; waiting for "stage 1 done: model_0030.pth" in mask_recipe_driver.log'
until grep -q 'stage 1 done: model_0030.pth' "$HOME/mask_recipe_driver.log" 2>/dev/null; do
  if ! pgrep -u "$USER" -f 'run_mask_recipe[.]sh' >/dev/null; then
    say 'ABORT: the mask driver ended before stage 1 finished; touching nothing'; exit 1
  fi
  sleep 10
done
say 'boundary reached; pretrain checkpoints:'
ls -l "$HOME/LEAD/lead/outputs/mask_lead_recipe/" >> "$LOG" 2>&1
stop() { local pids; pids=$(pgrep -u "$USER" -f "$1")
  if [ -z "$pids" ]; then say "  $3: none running"; return; fi
  say "  $3: kill -$2 "$pids; kill -"$2" $pids 2>/dev/null; }
kill_tree() { local c; for c in $(ps -o pid= --ppid "$1"); do kill_tree "$c"; done; kill -KILL "$1" 2>/dev/null; }
stop 'run_loader_benchmark[.]sh' TERM 'benchmark waiter'
stop 'run_final_chain3[.]sh'     TERM 'chain3'
stop 'run_mask_recipe[.]sh'      TERM 'mask driver'
sleep 2
for p in $(pgrep -u "$USER" -f 'training/train[.]py'); do say "  killing trainer tree rooted at $p"; kill_tree "$p"; done
sleep 5
for p in $(pgrep -u "$USER" -f 'training/train[.]py'); do say "  leftover trainer $p, killing its tree"; kill_tree "$p"; done
sleep 3
left=$(pgrep -u "$USER" -f 'training/train[.]py|run_mask_recipe[.]sh|run_final_chain3[.]sh|run_loader_benchmark[.]sh|run_evaluation[.]py' | wc -l)
say "our queue processes left: $left"
held=''
for p in $(pgrep -u "$USER"); do ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/nvidia && held="$held $p"; done
if [ -z "$held" ]; then say 'our processes holding /dev/nvidia*: none'
else say "our processes STILL holding /dev/nvidia*:$held"; for p in $held; do ps -o pid=,args= -p $p | cut -c1-120 >> "$LOG"; done; fi
say 'post-train dir now:'; ls -l "$HOME/LEAD/lead/outputs/mask_lead_recipe_post/" >> "$LOG" 2>&1
say 'DONE - ready for the module reload'
