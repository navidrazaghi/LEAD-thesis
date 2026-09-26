"""How much authority does attention have over what a token becomes?

Each block computes ``x = x + attn(ln1(x))``. Whatever modality mixture the
attention produces is added to a stream that already carries the token's own
content. If that addition is small next to the stream, then reweighting which
modality the attention reads changes little about what the token ends up being
-- which would explain why an eleven-fold shift in attention mass moves the
output only twofold.

This reports, per block, the norm of the attention's contribution against the
norm of the stream it is added to.
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
IMG, BEV = 12 * 36, 10 * 12

for entry in sys.argv[1:]:
    name, _, path = entry.partition("=")
    cfg, model = load_model(pathlib.Path(path), device)
    model.eval()
    rows, handles, stream = [], [], {}

    # The state is passed in rather than closed over. Every handle is removed
    # in the finally below, so a stale hook could not reach the next model's
    # state anyway -- but a hook that reads whatever the loop variable happens
    # to point at when it fires is a bug waiting for the day that stops being
    # true, and binding it here costs nothing.
    def pre(block, stream=stream):
        def hook(_m, ins, _o=None):
            stream[id(block)] = ins[0].detach().float()

        return hook

    def post(block, stream=stream, rows=rows):
        def hook(_m, _i, out):
            x = stream.get(id(block))
            if x is None:
                return
            a = out.detach().float()
            for lo, hi, tag in ((0, IMG, "image"), (IMG, IMG + BEV, "bev")):
                rows.append(
                    (
                        tag,
                        a[:, lo:hi].norm(dim=-1).mean().item(),
                        x[:, lo:hi].norm(dim=-1).mean().item(),
                    )
                )

        return hook

    for m in model.modules():
        if type(m).__name__ == "DeformableBlock":
            handles.append(m.register_forward_pre_hook(pre(m)))
            handles.append(m.attn.register_forward_hook(post(m)))
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
                if i >= 15:
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
    for tag in ("image", "bev"):
        sel = [(a, s) for t, a, s in rows if t == tag]
        a = sum(x for x, _ in sel) / len(sel)
        s = sum(y for _, y in sel) / len(sel)
        print(
            f"{name:8} {tag:6} attention output {a:8.3f}   residual stream {s:8.3f}   "
            f"attention is {100 * a / (a + s):5.1f}% of what leaves the block"
        )
