"""How much of the fusion does the camera actually carry?

Every measurement in the thesis reports a *shift* in the camera's share, never
its level, on the stated ground that levels are not comparable between models.
That is right for comparing rungs and wrong for the question here, which is not
"which rung shifts more" but "is there anything to shift". A gate can only move
reliance that exists. If the camera already carries almost none of the read,
then gating away from a failing camera has nothing to gain, and the asymmetry
the thesis reports -- the mechanism works under LiDAR damage and fails under
camera damage -- follows from the architecture rather than from the gate.

Two levels are reported per model, both under nominal sensors: the camera's
share of the attention mass, and its share of the output's sensitivity. The
second is the causal one.
"""

import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from analyze_gate import FusionProbe, load_model, to_device  # noqa: E402

device = torch.device("cuda:0")
for entry in sys.argv[1:]:
    name, _, path = entry.partition("=")
    cfg, model = load_model(pathlib.Path(path), device)
    model.eval()
    ds = model.build_dataset()
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(ds, "collate_fn", None),
        num_workers=4,
    )
    probe = FusionProbe(model)
    shares, sens = [], []
    try:
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= 30:
                    break
                batch = to_device(batch, device)
                with autocast(
                    device_type="cuda",
                    dtype=cfg.training.optimization.torch_dtype,
                    enabled=cfg.training.optimization.use_mixed_precision_training,
                ):
                    model(batch)
                s, _ = probe.read()
                if s is not None:
                    shares.append(s)
        # Causal share: replace each modality with same-scale noise and see how
        # far the predicted waypoints move.
        gen = torch.Generator(device=device)
        gen.manual_seed(20260820)
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= 30:
                    break
                batch = to_device(batch, device)
                with autocast(
                    device_type="cuda",
                    dtype=cfg.training.optimization.torch_dtype,
                    enabled=cfg.training.optimization.use_mixed_precision_training,
                ):
                    base = model(batch).future_waypoints.float()
                moves = {}
                for key, label in (("rgb", "cam"), ("rasterized_lidar", "lid")):
                    if key not in batch:
                        continue
                    keep = batch[key]
                    x = keep.float()
                    noise = (
                        torch.randn(
                            x.shape, generator=gen, device=device, dtype=x.dtype
                        )
                        * x.std()
                        + x.mean()
                    )
                    batch[key] = noise.to(keep.dtype)
                    with autocast(
                        device_type="cuda",
                        dtype=cfg.training.optimization.torch_dtype,
                        enabled=cfg.training.optimization.use_mixed_precision_training,
                    ):
                        out = model(batch).future_waypoints.float()
                    moves[label] = (out - base).norm(dim=-1).mean().item()
                    batch[key] = keep
                if len(moves) == 2 and sum(moves.values()) > 0:
                    sens.append(moves["cam"] / (moves["cam"] + moves["lid"]))
    finally:
        probe.close()
    a = sum(shares) / len(shares) if shares else float("nan")
    b = sum(sens) / len(sens) if sens else float("nan")
    print(
        f"{name:8}  camera share of attention mass {a:.3f}   camera share of sensitivity {b:.3f}"
    )
