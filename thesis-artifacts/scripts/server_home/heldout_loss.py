"""Training objective of a finished checkpoint on the 28 held-out logs.

Training logs only its own loss, on data it is fitting; a falling curve says the
optimiser works, not that the model generalises. This measures the same
quantity on logs no run trained on (~/new_subset/held_out_28.txt), so train and
held-out loss can be read side by side.

Same config as the run -- initialize_config merges the checkpoint's own
config.yaml -- with only what would make the number noisy or unfair switched
off: rig perturbation, colour augmentation, sensor degradation, and
torch.compile. Frames are computed from the logs rather than the cache store,
so every model is scored through the identical path. Loss weights are the ones
the run ended on (epoch 30), normalised as the trainer normalises them, and the
forward runs under bf16 autocast as training does. No gradients, eval mode.

Usage: python heldout_loss.py NAME CHECKPOINT [--batches N]
Appends one row per loss term plus "objective" to results/heldout_loss.csv.
"""
import csv
import os
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path.home() / "LEAD/lead"
os.chdir(ROOT)
name, checkpoint = sys.argv[1], str(pathlib.Path(sys.argv[2]).resolve())
max_batches = int(sys.argv[sys.argv.index("--batches") + 1]) if "--batches" in sys.argv else None
# --logs swaps in another list, e.g. training logs as the control that shows
# this path reproduces the trainer's own number on data it fitted.
logs_file = sys.argv[sys.argv.index("--logs") + 1] if "--logs" in sys.argv else str(pathlib.Path.home() / "new_subset/held_out_28.txt")
held = [line.split("/")[1] for line in pathlib.Path(logs_file).read_text().split()]
assert len(held) == 28, len(held)
name = f"{name}@{pathlib.Path(logs_file).stem}"

sys.argv = [sys.argv[0],
            f"training.experiment.initial_weights_file={checkpoint}",
            "training.experiment.resume_from_last_checkpoint=true",   # strict weight load
            "training.data.use_sensor_perturbation=false",
            "training.data.use_color_augmentation=false",
            "training.data.use_sensor_degradation=false",
            "training.data.read_from_cache_store=false",
            "training.optimization.use_torch_compile=false",
            f"training.data.py123d_log_names=[{','.join(held)}]"]

from lead.training import run_config  # noqa: E402
from lead.training.train import LeadLightningModule  # noqa: E402

config = run_config.initialize_config()
module = LeadLightningModule(config)
model = getattr(module, "_raw_model", module.model).cuda().eval()
weights = model.per_task_loss_weights(30)
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

dataset = model.build_dataset()
loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False,
                    collate_fn=getattr(dataset, "collate_fn", None), num_workers=16)
sums, samples, start = {}, 0, time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["current_gradient_step"] = 0
        size = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() > 0)
        losses, _ = model.compute_loss(model(batch), batch)
        for key, value in losses.items():
            if key in weights:
                sums[key] = sums.get(key, 0.0) + weights[key] * float(value) * size
        samples += size

terms = {k: v / samples for k, v in sums.items()}
terms["objective"] = sum(terms.values())
print(f"{name}: {samples} held-out samples from {len(dataset)} in {time.time() - start:.0f} s")
for key, value in sorted(terms.items()):
    print(f"  {key:28s} {value:.6f}")
out = ROOT / "results/heldout_loss.csv"
new = not out.exists()
with out.open("a", newline="") as handle:
    writer = csv.writer(handle)
    if new:
        writer.writerow(["model", "checkpoint", "samples", "term", "value"])
    for key, value in sorted(terms.items()):
        writer.writerow([name, checkpoint, samples, key, round(value, 6)])
