#!/bin/bash
# Waits for the EGCA training job to exit, then runs the LEAD closed-loop
# evaluation on a free GPU. Designed to survive ssh disconnects (run detached).

TRAIN_PID=4189943
LOG=~/auto_eval.log
ROUTE=src/lead/routes/benchmark_routes/bench2drive/23687.xml
TM_PORT=8055
CARLA_PORT=2000

log() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }

log "=== watcher started; waiting for training PID $TRAIN_PID to exit ==="

# 1. Wait for the training job to finish.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 120
done
log "training PID $TRAIN_PID exited"

# Let the driver reclaim GPU memory.
sleep 90
log "GPU state after training exit:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader >>"$LOG" 2>&1

# 2. Make sure no stale CARLA is holding the port.
OLD=$(pgrep -f "CarlaUE4-Linux-Shipping" | head -1)
if [ -n "$OLD" ]; then
    log "killing stale CARLA $OLD"
    kill -9 "$OLD" 2>/dev/null
    sleep 10
fi

# 3. Start CARLA detached.
log "starting CARLA"
cd ~/CARLA/standard_0916 || { log "FATAL: CARLA dir missing"; exit 1; }
setsid nohup ./CarlaUE4.sh -world-port=$CARLA_PORT -nosound -RenderOffScreen \
    -carla-streaming-port=$((CARLA_PORT + 1)) >~/carla_server_auto.log 2>&1 </dev/null &
disown

# 4. Wait for the RPC port (up to 10 minutes).
for i in $(seq 1 40); do
    if ss -ltn | grep -q ":$CARLA_PORT "; then
        log "CARLA port open after $((i * 15))s"
        break
    fi
    sleep 15
done
if ! ss -ltn | grep -q ":$CARLA_PORT "; then
    log "FATAL: CARLA never opened port $CARLA_PORT"
    tail -20 ~/carla_server_auto.log >>"$LOG"
    exit 1
fi
sleep 20

# 5. Run the evaluation.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lead
export TIMM_USE_OLD_CACHE=1
cd ~/LEAD/lead || { log "FATAL: LEAD dir missing"; exit 1; }
export PATH="$PWD/scripts/cli:$PATH"

log "starting evaluation on $ROUTE"
python -m lead \
    --checkpoint checkpoints/transfuser \
    --routes "$ROUTE" \
    --bench2drive \
    --traffic-manager-port $TM_PORT \
    --timeout 600 >~/eval_auto.log 2>&1
RC=$?
log "evaluation finished with exit code $RC"

# 6. Stop CARLA so it does not sit on the GPU afterwards.
CP=$(pgrep -f "CarlaUE4-Linux-Shipping" | head -1)
[ -n "$CP" ] && kill -9 "$CP" 2>/dev/null && log "stopped CARLA $CP"

log "=== results ==="
ls -la ~/LEAD/lead/outputs/local_evaluation/23687/ >>"$LOG" 2>&1
log "AUTO_EVAL_DONE"
