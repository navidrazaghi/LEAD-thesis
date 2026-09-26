"""Do the two modalities arrive at the fusion on comparable scales?

Chapter three decomposes the output's sensitivity into a value path and a
weight path. The gate acts only on the weight path. The measurements show
attention mass on the camera at 0.42 while its causal share is 0.26, which is
what you would see if the value path favoured LiDAR regardless of how the
weights are set.

If the LiDAR tokens simply arrive with larger magnitude, then a query that
splits its attention evenly still reads mostly LiDAR, and no amount of gating
fixes that. This measures the magnitude of the tokens of each modality at the
point the fusion reads them.
"""

import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from analyze_gate import load_model, to_device  # noqa: E402

device = torch.device("cuda:0")
IMAGE_TOKENS = 12 * 36  # the image anchor grid
BEV_TOKENS = 10 * 12  # the bird's-eye-view grid

for entry in sys.argv[1:]:
    name, _, path = entry.partition("=")
    cfg, model = load_model(pathlib.Path(path), device)
    model.eval()

    # Hook whatever the deformable blocks receive, before they normalise it.
    seen = []
    handles = []
    for m in model.modules():
        if type(m).__name__ == "DeformableBlock":
            handles.append(
                m.register_forward_pre_hook(
                    lambda _m, inputs, sink=seen: sink.append(inputs[0].detach())
                )
            )
    ds = model.build_dataset()
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(ds, "collate_fn", None),
        num_workers=4,
    )
    cam, lid = [], []
    try:
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= 20:
                    break
                seen.clear()
                batch = to_device(batch, device)
                with autocast(
                    device_type="cuda",
                    dtype=cfg.training.optimization.torch_dtype,
                    enabled=cfg.training.optimization.use_mixed_precision_training,
                ):
                    model(batch)
                for x in seen:
                    if x.shape[1] < IMAGE_TOKENS + BEV_TOKENS:
                        continue
                    f = x.float()
                    cam.append(f[:, :IMAGE_TOKENS].norm(dim=-1).mean().item())
                    lid.append(
                        f[:, IMAGE_TOKENS : IMAGE_TOKENS + BEV_TOKENS]
                        .norm(dim=-1)
                        .mean()
                        .item()
                    )
    finally:
        for h in handles:
            h.remove()
    if not cam:
        print(f"{name:8}  no deformable blocks (dense baseline)")
        continue
    c, bev = sum(cam) / len(cam), sum(lid) / len(lid)
    print(
        f"{name:8}  mean token norm  image {c:8.3f}   BEV {bev:8.3f}   ratio BEV/image {bev / c:6.2f}"
    )
