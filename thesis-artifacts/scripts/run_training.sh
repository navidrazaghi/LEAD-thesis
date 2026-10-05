#!/usr/bin/env bash
# Sequential training queue.  Every configuration gets the identical budget
# (20 epochs, batch 64) so the ablation table compares mechanisms, not budgets.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egca
cd ~/thesis/code
COMMON="train.epochs=20 train.batch_size=64 train.num_workers=16"
run () {  # name seed extra-overrides...
  local name=$1 seed=$2; shift 2
  if [ -f "checkpoints/$name/best.pth" ]; then
    echo "$(date +%F\ %T) SKIP $name (already trained)"; return
  fi
  echo "$(date +%F\ %T) START $name seed=$seed $*"
  python -u -m egca.training.train --config configs/egca.yaml --seed "$seed" \
      --set $COMMON train.ckpt_dir=checkpoints/$name "$@" \
      2>&1 | tee ~/logs/train_$name.log
  echo "$(date +%F\ %T) END   $name"
}
run egca_s0 0
run egca_s1 1
run egca_s2 2
run full_attn 0 model.fusion.attention=full
run no_gate   0 model.fusion.gate=false
run no_dropout 0 train.sensor_dropout=0.0
run camera_only 0 model.fusion.mode=camera_only
run lidar_only  0 model.fusion.mode=lidar_only
run concat      0 model.fusion.mode=concat
run no_aux      0 model.aux.bev_seg=false model.aux.depth=false
run late        0 model.fusion.mode=late
echo "$(date +%F\ %T) ALL TRAINING DONE"
