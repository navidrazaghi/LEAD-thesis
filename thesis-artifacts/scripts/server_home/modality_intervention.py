# -*- coding: utf-8 -*-
"""Does the model's plan move when a sensor is destroyed? Same frames, three inputs.

Closed loop showed the deformable run on the diverse subset scoring *identically*
with and without LiDAR on six imperfect routes -- the signature the old baseline
left under camera loss, when it was later shown not to read the camera at all.
Two explanations fit, and they point opposite ways:

  (a) the model does not use LiDAR, so "robust to LiDAR loss" is really "never
      relied on it";
  (b) the LiDAR damage never reaches this model's input, which is a plumbing bug.

This separates them. It first checks that the damage actually changed the input
tensor (rules (b) in or out), then measures how far the predicted waypoints move
when each modality is destroyed, for this model and for the baseline trained on
the same data. The damage is the evaluation's own ``degrade_batch`` at severity
1.0, seeded, so both models see identical corrupted frames.

Usage: python modality_intervention.py NAME=CHECKPOINT_DIR [...]
"""
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/common"))
from analyze_gate import load_model, to_device  # noqa: E402
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

BATCHES, BATCH_SIZE = 40, 8
device = torch.device("cuda:0")


def plan(model, batch):
    """Predicted future waypoints for a batch, under the training precision."""
    batch = dict(batch)
    batch["current_gradient_step"] = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return model(batch).future_waypoints.float()


for entry in sys.argv[1:]:
    name, _, path = entry.partition("=")
    _, model = load_model(pathlib.Path(path), device)
    model.eval()
    dataset = model.build_dataset()
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        collate_fn=getattr(dataset, "collate_fn", None), num_workers=8,
        generator=torch.Generator().manual_seed(0),
    )
    moved = {"lidar": [], "camera": []}
    input_changed = {"lidar": [], "camera": []}
    for index, batch in enumerate(loader):
        if index >= BATCHES:
            break
        batch = to_device(batch, device)
        intact = plan(model, batch)
        for modality, key in (("lidar", "rasterized_lidar"), ("camera", "rgb")):
            damaged = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            damaged = degrade_batch(damaged, modality, 1.0, generator=torch.Generator(device=device).manual_seed(index))
            input_changed[modality].append(float((damaged[key].float() - batch[key].float()).abs().mean()))
            out = plan(model, damaged)
            # Mean Euclidean displacement over the 8 future points, per sample.
            moved[modality].append((out - intact).norm(dim=-1).mean(dim=-1).cpu())
    for modality in ("lidar", "camera"):
        shift = torch.cat(moved[modality])
        change = sum(input_changed[modality]) / len(input_changed[modality])
        print(f"{name:20s} destroy {modality:6s}: input mean |change| {change:8.4f} | "
              f"waypoints move mean {float(shift.mean()):.3f} m, median {float(shift.median()):.3f} m, "
              f"share of samples moving < 1 cm {float((shift < 0.01).float().mean()):.2f}  (n={shift.numel()})")
    del model
    torch.cuda.empty_cache()
