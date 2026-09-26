"""What does the observability supervision actually look like?

If the gate has to learn a relation from these labels, the labels have to vary.
This reports the distribution of the per-token targets and, more to the point,
of the log ratio between the two modalities -- because that ratio is the whole
signal the gate is asked to reproduce.
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
cfg, model = load_model(pathlib.Path("outputs/rung4_light_auxiliary_post"), device)
model.eval()
tm = next(m for m in model.modules() if type(m).__name__ == "ObservabilityTokenTargets")
ds = model.build_dataset()
loader = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    drop_last=True,
    collate_fn=getattr(ds, "collate_fn", None),
    num_workers=4,
)
vals, ratios, both = [], [], 0
with torch.inference_mode():
    for i, batch in enumerate(loader):
        if i >= 60:
            break
        batch = to_device(batch, device)
        with autocast(
            device_type="cuda",
            dtype=cfg.training.optimization.torch_dtype,
            enabled=cfg.training.optimization.use_mixed_precision_training,
        ):
            model(batch)
        t, m = tm(batch["observability"].float(), batch["observability_mask"].float())
        t, m = t.float(), m.float()
        sup = m > 0
        vals.append(t[sup].cpu())
        pair = (m[..., 0] > 0) & (m[..., 1] > 0)
        both += int(pair.sum())
        vc, vl = t[..., 0][pair], t[..., 1][pair]
        ok = (vc > 1e-6) & (vl > 1e-6)
        ratios.append((vc[ok].log() - vl[ok].log()).cpu())

v = torch.cat(vals)
r = torch.cat(ratios)
print("supervised token/modality entries :", v.numel())
print("tokens supervised in BOTH modalities:", both)
print()
print("distribution of the observability target:")
for q in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
    print(f"   q{q:<5} {v.quantile(q).item():.4f}")
print(f"   fraction == 0     : {(v <= 1e-6).float().mean().item():.3f}")
print(f"   fraction >= 0.95  : {(v >= 0.95).float().mean().item():.3f}")
print(f"   fraction in (0,1) : {((v > 1e-6) & (v < 0.999)).float().mean().item():.3f}")
print()
print("distribution of log(V_cam / V_lid) -- the signal the gate must reproduce:")
print(f"   n={r.numel()}  sd={r.std().item():.4f}")
for q in (0.05, 0.25, 0.50, 0.75, 0.95):
    print(f"   q{q:<5} {r.quantile(q).item():+.4f}")
print(f"   fraction |ratio| < 0.05 : {(r.abs() < 0.05).float().mean().item():.3f}")
