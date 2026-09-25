"""Per-frame open-loop waypoint error, so the table can carry an interval.

The open-loop table reports one number per rung per condition and nothing about
how precisely that number is known. The closed-loop table, by contrast, carries
a paired interval for every comparison. That asymmetry is not a matter of
principle -- the frames are there, they are the same frames for every rung in
the same order, and the pairing that makes the closed-loop analysis work
applies here unchanged.

So this dumps the error of every frame rather than their mean. The analysis
that follows is the chapter's own: differences taken frame by frame against a
reference rung, a t interval on the mean difference, and an effect size.

What this does not measure has to stay in view. Each rung is one training run,
so an interval computed here describes how precisely the mean error of *this
trained model* is known over frames. It says nothing about how much that mean
would move if the rung were trained again with another seed. The two are
different quantities and the chapter says so where it reports them.
"""

import argparse
import csv
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

_DAMAGE_SEED = 20260820


def per_frame(model, lead_config, loader, condition, batches, device):
    """Waypoint L2 of every frame, in loader order.

    Args:
        model: Loaded policy in eval mode.
        lead_config: Config it was trained with.
        loader: Loader over the probe frames.
        condition: ``(modality, severity)``.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        A list of per-frame errors.
    """
    modality, severity = condition
    generator = torch.Generator(device=device)
    generator.manual_seed(_DAMAGE_SEED)
    errors: list[float] = []
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            if seen >= batches:
                break
            batch = to_device(batch, device)
            batch = degrade_batch(batch, modality, severity, generator)
            with autocast(
                device_type="cuda",
                dtype=lead_config.training.optimization.torch_dtype,
                enabled=(
                    lead_config.training.optimization.use_mixed_precision_training
                ),
            ):
                prediction = model(batch)
            predicted = getattr(prediction, "future_waypoints", None)
            if predicted is None or "future_waypoints" not in batch:
                seen += 1
                continue
            label = batch["future_waypoints"].float()
            predicted = predicted.float()
            horizon = min(predicted.shape[1], label.shape[1])
            # Mean over the waypoints of one frame, kept per frame.
            frame = (
                (predicted[:, :horizon] - label[:, :horizon]).norm(dim=-1).mean(dim=-1)
            )
            errors.extend(frame.cpu().tolist())
            seen += 1
    return errors


def main() -> None:
    """Dump per-frame errors for every model and condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "results/openloop_frames.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    rows = []
    for entry in args.models:
        name, _, path = entry.partition("=")
        print(f"\n=== {name} ===", flush=True)
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        dataset = model.build_dataset()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        for pair in args.conditions:
            modality, _, severity = pair.partition(":")
            errors = per_frame(
                model, lead_config, loader,
                (modality, float(severity)), args.batches, device,
            )
            mean = sum(errors) / max(len(errors), 1)
            print(f"  {pair:14} frames={len(errors):5}  mean={mean:.4f}", flush=True)
            for index, value in enumerate(errors):
                rows.append({"model": name, "condition": pair,
                             "frame": index, "l2": value})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "condition",
                                                    "frame", "l2"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
