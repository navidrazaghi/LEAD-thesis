"""Does the modality imbalance survive the block's own LayerNorm?

The earlier probe measured the tokens arriving at each deformable block and
found the BEV group 1.75 times larger than the image group in rung 2a. That was
read as an imbalance the gate cannot fix. But the block normalises its input
before the operator reads it, and LayerNorm acts on each token independently,
so it may erase the difference before the values are ever computed.

This measures the same two groups on both sides of that normalisation. If the
ratio after it is one, the imbalance never reaches the value path and any
per-modality normalisation added on top would be a no-op.
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
    before, after, values = [], [], []
    handles = []
    for m in model.modules():
        if type(m).__name__ != "DeformableBlock":
            continue
        handles.append(
            m.register_forward_pre_hook(
                lambda _m, ins, s=before: s.append(ins[0].detach())
            )
        )
        handles.append(
            m.ln1.register_forward_hook(
                lambda _m, _i, out, s=after: s.append(out.detach())
            )
        )
        handles.append(
            m.attn.value_proj.register_forward_hook(
                lambda _m, _i, out, s=values: s.append(out.detach())
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

    def ratio(sink):
        r = []
        for x in sink:
            if x.shape[1] < IMG + BEV:
                continue
            f = x.float()
            c = f[:, :IMG].norm(dim=-1).mean().item()
            bev = f[:, IMG : IMG + BEV].norm(dim=-1).mean().item()
            if c > 0:
                r.append((c, bev))
        return r

    try:
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                if i >= 15:
                    break
                before.clear()
                after.clear()
                values.clear()
                batch = to_device(batch, device)
                with autocast(
                    device_type="cuda",
                    dtype=cfg.training.optimization.torch_dtype,
                    enabled=cfg.training.optimization.use_mixed_precision_training,
                ):
                    model(batch)
                if i == 0:
                    stages = {
                        "block input": ratio(before),
                        "after ln1": ratio(after),
                        "after value_proj": ratio(values),
                    }
                    print(f"\n{name}")
                    for label, r in stages.items():
                        if not r:
                            print(f"  {label:18} (not captured)")
                            continue
                        c = sum(a for a, _ in r) / len(r)
                        bev = sum(b for _, b in r) / len(r)
                        print(
                            f"  {label:18} image {c:8.3f}  BEV {bev:8.3f}  ratio {bev / c:5.2f}"
                        )
                    break
    finally:
        for h in handles:
            h.remove()
