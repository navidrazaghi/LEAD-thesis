#!/bin/bash
# Rung 5: rung 4 plus a per-token gain on the attention's residual contribution.
#
# One change against rung 4, as every rung in this ladder is. The gate already
# decides which modality a query reads from and has pushed the camera's share of
# a BEV query's attention mass to about half. What it cannot decide is how much
# of that read survives into the token, because the block adds the attention
# output to a residual stream the gate does not touch -- measured at roughly
# 0.40 of what leaves each block. The product of those two factors predicts the
# causal camera reliance for both rung 2a and rung 4, which is what identifies
# the second factor as the ceiling.
#
# This rung supplies control over that second factor and nothing else. The gain
# is zero-initialised, verified numerically to leave the block unchanged at
# step zero, and adds 65 parameters per block against 48,034.
#
# Stated plainly for the write-up: this rung was designed after seeing the
# results it responds to, so it is exploratory and not part of the ladder that
# was fixed before the closed-loop campaign ran.
#
# Waits for the GPU rather than racing whatever is on it.

set -u
cd ~/LEAD/lead || exit 1
ulimit -n 65536
export TIMM_USE_OLD_CACHE=1

echo "[$(date +%H:%M:%S)] waiting for the weather evaluation to finish"
while ! grep -q "weather evaluation done" ~/weather_eval.log 2>/dev/null; do
  sleep 300
done
echo "[$(date +%H:%M:%S)] it finished; settling before taking the GPU"
sleep 300
while pgrep -f "Carla[U]E4-Linux" > /dev/null 2>&1; do
  sleep 60
done

DEFORMABLE=lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone

echo "[$(date +%H:%M:%S)] training rung5"
bash scripts/common/run_rung.sh rung5_residual_gain \
  policy.transfuser.backbone_target="$DEFORMABLE" \
  policy.transfuser.deformable_calibrated_reference=true \
  policy.transfuser.use_observability=true \
  policy.transfuser.use_observability_gate=true \
  policy.transfuser.use_residual_gain=true \
  policy.transfuser.observability_loss_weight=0.2 \
  policy.transfuser.observability_gate_loss_weight=0.2 \
  training.data.use_sensor_degradation=true \
  || { echo "rung5 training failed"; exit 1; }

echo "[$(date +%H:%M:%S)] measuring what it changed, before spending CARLA time on it"
export LEAD_RUNTIME_TYPE_CHECKING=false
PY=~/miniconda3/envs/lead/bin/python

$PY scripts/common/light_experiments.py intervention \
  --models rung4=outputs/rung4_light_auxiliary_post \
           rung5=outputs/rung5_residual_gain_post \
  --batches 60 --batch-size 8 \
  --out results/intervention_rung5.csv

$PY scripts/common/query_split.py rung5=outputs/rung5_residual_gain_post
$PY scripts/common/residual_share.py rung5=outputs/rung5_residual_gain_post

echo "[$(date +%H:%M:%S)] done; read the causal reliance before deciding on closed loop"
