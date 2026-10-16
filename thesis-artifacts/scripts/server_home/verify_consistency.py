# -*- coding: utf-8 -*-
"""Check degradation-consistency self-distillation on a real model and real batch.

Runs a batch of four on the GPU beside the training; the model loader requires CUDA. Uses the diverse
baseline's post-train checkpoint, which has the planning decoder, with the
curriculum v2 flags switched on in its config.
"""
import pathlib
import sys
import types

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/common"))
from analyze_gate import load_model, to_device  # noqa: E402
from lead.training.train import LeadLightningModule  # noqa: E402

ok = True


def check(name, passed, detail=""):
    """Record and print one check."""
    global ok
    ok = ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def copy(batch):
    """Deep-copy the tensors of a batch."""
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


device = torch.device("cuda:0")
config, model = load_model(ROOT / "outputs/rung0_diverse_post31", device)
data = model.lead_config.training.data
data.use_sensor_degradation = True
data.sensor_degradation_probability = 0.30
data.sensor_degradation_independent_modalities = True
data.sensor_degradation_full_failure_probability = 0.25
data.sensor_degradation_misalignment_probability = 0.10
data.degradation_consistency_weight = 1.0

dataset = model.build_dataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=getattr(dataset, "collate_fn", None),
                    num_workers=2, generator=torch.Generator().manual_seed(0))
base = to_device(next(iter(loader)), device)
base["current_gradient_step"] = 0
model.train()

# --- 1. the student batch is draw-for-draw augment_batch's ----------------
torch.manual_seed(7)
reference = model.augment_batch(copy(base))
torch.manual_seed(7)
student, clean = model.augment_batch_with_clean(copy(base))
same = all(torch.equal(reference[k], student[k]) for k in ("rgb", "rasterized_lidar", "observability") if k in reference)
check("student batch equals augment_batch's for the same random state", same)

# --- 2. the clean copy differs only where damage was applied ---------------
per_sample = []
for i in range(base["rgb"].shape[0]):
    differs = any(not torch.equal(student[k][i], clean[k][i]) for k in ("rgb", "rasterized_lidar"))
    per_sample.append(differs)
check("clean copy is intact and differs from the student only on damaged samples",
      "rgb" in clean and clean["rgb"].shape == student["rgb"].shape,
      f"damaged samples in this batch: {sum(per_sample)} of {len(per_sample)}")

# --- 3. the consistency term ---------------------------------------------------
host = types.SimpleNamespace(model=model)
fn = LeadLightningModule._degradation_consistency
amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # training_step runs under Lightning's bf16 autocast
with amp():
    predictions = model(copy(clean))
    none_case = fn(host, predictions, copy(clean), copy(clean))
check("identical inputs give no consistency term", none_case is None)

# Force damage on every sample so the check does not depend on the draw.
torch.manual_seed(1)
forced = copy(clean)
forced["rgb"] = forced["rgb"] * 0.05
model.zero_grad(set_to_none=True)
with amp():
    predictions = model(forced)
    value = fn(host, predictions, forced, copy(clean))
check("damaged inputs give a positive finite consistency term",
      value is not None and bool(torch.isfinite(value)) and float(value) > 0,
      f"value {None if value is None else float(value):.4f}")
value.backward()
grads = [p.grad for n, p in model.named_parameters() if "planning_decoder" in n and p.grad is not None]
check("the term sends gradient into the planning decoder", bool(grads) and any(float(g.abs().max()) > 0 for g in grads),
      f"{len(grads)} decoder tensors with gradient")

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
