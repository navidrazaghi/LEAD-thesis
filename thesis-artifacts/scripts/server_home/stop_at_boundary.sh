#!/bin/bash
# Stop our own queue at the mask run's pretrain -> post-train boundary, so the
# NVIDIA kernel module can be reloaded (option A, approved by the user). The
# trigger is the driver's own 'stage 1 done' line, printed only after the
# pretrain process has exited and its checkpoint was found -- never a clock.
LOG=$HOME/boundary_stop.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
say 'armed; waiting for "stage 1 done: model_0030.pth" in mask_recipe_driver.log'
until grep -q 'stage 1 done: model_0030.pth' "$HOME/mask_recipe_driver.log" 2>/dev/null; do
  if ! pgrep -u "$USER" -f 'run_mask_recipe[.]sh' >/dev/null; then
    say 'ABORT: run_mask_recipe.sh ended before stage 1 finished; touching nothing'
    exit 1
  fi
  sleep 10
done
say 'boundary reached; pretrain checkpoints:'
ls -l "$HOME/LEAD/lead/outputs/mask_lead_recipe/" >> "$LOG" 2>&1
stop() {
  local pids; pids=$(pgrep -u "$USER" -f "$1")
  if [ -z "$pids" ]; then say "  $3: none running"; return; fi
  say "  $3: kill -$2 "$pids; kill -"$2" $pids 2>/dev/null
}
stop 'run_loader_benchmark[.]sh' TERM 'benchmark waiter'
stop 'run_final_chain3[.]sh'     TERM 'chain3'
stop 'run_mask_recipe[.]sh'      TERM 'mask driver'
sleep 2
stop 'training/train[.]py'       KILL 'stage-2 trainer and loader workers'
sleep 5
stop 'training/train[.]py'       KILL 'trainer leftovers'
left=$(pgrep -u "$USER" -f 'training/train[.]py|run_mask_recipe[.]sh|run_final_chain3[.]sh|run_loader_benchmark[.]sh|run_evaluation[.]py' | wc -l)
say "our queue processes left: $left"
held=''
for p in $(pgrep -u "$USER"); do ls -l /proc/$p/fd 2>/dev/null | grep -q /dev/nvidia && held="$held $p"; done
say "our processes holding /dev/nvidia*: ${held:-none}"
say 'post-train dir now:'; ls -l "$HOME/LEAD/lead/outputs/mask_lead_recipe_post/" >> "$LOG" 2>&1
say 'DONE - ready for the module reload'
