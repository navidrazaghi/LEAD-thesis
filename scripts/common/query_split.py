"""Which queries does the gate actually influence, and do those queries matter?

The planning decoder reads only the BEV features. Image tokens never reach it
directly, so what the camera contributes to driving is exactly what the fusion
writes into the BEV tokens -- nothing else.

The gate, though, biases every query: the 432 image-grid queries as well as the
120 BEV-grid queries. The camera share of attention mass reported in the thesis
is an average over all of them. If the image queries are the ones reading from
the camera, most of the gate's influence lands on queries whose output the
decoder never sees.

This splits the same quantity by query group.
"""

import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))
from analyze_gate import load_model, to_device  # noqa: E402

device = torch.device("cuda:0")
IMG, BEV = 12 * 36, 10 * 12
CAMERA_LEVEL = 0

for entry in sys.argv[1:]:
    name, _, path = entry.partition("=")
    cfg, model = load_model(pathlib.Path(path), device)
    model.eval()
    rows = []
    handles = []

    # Bound rather than closed over: every handle is removed before the next
    # model is built, so a stale hook could not reach this list anyway, but a
    # hook that appends to whatever the loop variable points at when it fires
    # is a bug waiting for the day that stops being true.
    def make_hook(block, rows=rows):
        def hook(_m, inputs, _out):
            x = inputs[0].detach().float()
            a = block.attn
            b, t, _ = x.shape
            logits = a.attention_weights(x).view(
                b, t, a.n_head, a.num_levels, a.num_points
            )
            g = block.gate(x).detach().float() if block.gate is not None else None
            if g is not None:
                logits = logits + g.view(b, t, 1, a.num_levels, 1)
            w = F.softmax(logits.flatten(-2), dim=-1).view_as(logits)
            share = w[..., CAMERA_LEVEL, :].sum(-1).mean(2)  # [B, T]
            rows.append(
                (share[:, :IMG].mean().item(), share[:, IMG : IMG + BEV].mean().item())
            )

        return hook

    for m in model.modules():
        if type(m).__name__ == "DeformableBlock":
            handles.append(m.register_forward_hook(make_hook(m)))
    ds = model.build_dataset()
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(ds, "collate_fn", None),
        num_workers=4,
    )
    try:
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= 20:
                    break
                batch = to_device(batch, device)
                with autocast(
                    device_type="cuda",
                    dtype=cfg.training.optimization.torch_dtype,
                    enabled=cfg.training.optimization.use_mixed_precision_training,
                ):
                    model(batch)
    finally:
        for h in handles:
            h.remove()
    if not rows:
        print(f"{name:8}  no deformable blocks")
        continue
    qi = sum(a for a, _ in rows) / len(rows)
    qb = sum(b for _, b in rows) / len(rows)
    overall = (qi * IMG + qb * BEV) / (IMG + BEV)
    print(
        f"{name:8}  camera mass read by IMAGE queries {qi:.3f}   "
        f"by BEV queries {qb:.3f}   all queries {overall:.3f}"
    )
