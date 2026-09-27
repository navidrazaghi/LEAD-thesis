"""How far a caution signal moves, measured on cached frames rather than argued.

The governor's regime turned out to be one nothing drives in, so what a signal
does under each condition is now the cheapest thing worth knowing about it: it
costs minutes here against hours of simulator, and it is what says whether a
signal has anything to report before a night is spent finding out.

Two signals are swept here, the two that return a caution directly:

``observability``
    The trained head, combined across modalities by the configured rule. Its
    range is already known from the recorded per-modality means, and
    ``caution_signal_range.py`` reports that without running the model at all.
    It is included here so the two can be read side by side on identical frames
    and identical damage.

``cross_modal``
    The camera's predicted depth against the LiDAR's returns. This one has never
    been measured on real data -- it has unit tests on synthetic tensors and
    nothing else -- and it is the signal with the most to prove, because it is
    the one claimed to see what the observability head is blind to. The head
    reports per modality and the governor takes the better of the two, so a
    single destroyed sensor leaves it correctly silent. A comparison between the
    two sensors should not be silent there.

The ensemble keeps its own script, ``ensemble_spread_range.py``, because what it
produces is a spread in meters that a baseline and a scale then map, and both of
those have to be derived from the same measurement.
"""

import argparse
import csv
import pathlib
import statistics
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402

from lead.evaluation.inference.caution import (  # noqa: E402
    cross_modal_caution,
    observability_caution,
)
from lead.policy.transfuser.utils.sensor_degradation import (  # noqa: E402
    degrade_batch,
)

# Fixed, so two checkpoints meet identical damage on identical frames.
_DAMAGE_SEED = 20260827

_CONDITIONS = (
    ("none", 0.0),
    ("camera", 0.5),
    ("camera", 1.0),
    ("lidar", 0.5),
    ("lidar", 1.0),
    ("both", 0.5),
    ("both", 1.0),
)


def caution_under(model, lead_config, loader, condition, signal, batches, device):
    """Per-batch caution under one condition.

    Args:
        model: Loaded policy in evaluation mode.
        lead_config: The config it was trained with.
        loader: Loader over the probe frames.
        condition: ``(modality, severity)``.
        signal: ``"observability"`` or ``"cross_modal"``.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        One caution per batch.

    Raises:
        SystemExit: If the model does not produce what the signal needs, which
            means it was built without that head rather than that the head had
            nothing to say.
    """
    modality, severity = condition
    generator = torch.Generator(device=device)
    generator.manual_seed(_DAMAGE_SEED)
    cautions: list[float] = []
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if seen >= batches:
                break
            batch = to_device(batch, device)
            batch = degrade_batch(batch, modality, severity, generator)
            with autocast(
                device_type=device.type,
                dtype=lead_config.training.optimization.torch_dtype,
                enabled=(
                    device.type == "cuda"
                    and lead_config.training.optimization.use_mixed_precision_training
                ),
            ):
                prediction = model(batch)

            if signal == "observability":
                if prediction.observability is None:
                    raise SystemExit(
                        "this checkpoint has no observability head; there is "
                        "nothing to measure for that signal.",
                    )
                cautions.append(
                    observability_caution(prediction.observability, lead_config),
                )
            else:
                if prediction.depth is None or "rasterized_lidar" not in batch:
                    raise SystemExit(
                        "this checkpoint has no depth head, or the batch has no "
                        "LiDAR raster; the cross-modal check needs both.",
                    )
                cautions.append(
                    cross_modal_caution(
                        prediction.depth,
                        batch["rasterized_lidar"],
                        lead_config,
                    ),
                )
            seen += 1
    return cautions


def main() -> int:
    """Sweep every condition and report the swing from intact.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument(
        "--signal",
        choices=("observability", "cross_modal"),
        required=True,
    )
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results" / "caution_forward_sweep.csv",
    )
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    rows = []
    for pair in arguments.models:
        name, _, path = pair.partition("=")
        lead_config, model = load_model(pathlib.Path(path), device)
        dataset = model.build_dataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=arguments.workers,
        )

        print(f"\n{name}  signal={arguments.signal}")
        print(f"  {'condition':<14}{'caution':>10}{'sd':>9}{'batches':>9}")
        intact = None
        for modality, severity in _CONDITIONS:
            cautions = caution_under(
                model,
                lead_config,
                loader,
                (modality, severity),
                arguments.signal,
                arguments.batches,
                device,
            )
            if not cautions:
                continue
            mean = statistics.mean(cautions)
            spread = statistics.stdev(cautions) if len(cautions) > 1 else 0.0
            label = f"{modality}:{severity}"
            print(f"  {label:<14}{mean:10.3f}{spread:9.3f}{len(cautions):9d}")
            rows.append(
                {
                    "model": name,
                    "signal": arguments.signal,
                    "condition": label,
                    "caution": f"{mean:.6f}",
                    "sd": f"{spread:.6f}",
                    "batches": len(cautions),
                },
            )
            if modality == "none":
                intact = mean

        if intact is not None:
            print(f"  {'--- swing from intact ---':<14}")
            for row in rows:
                if row["model"] != name or row["condition"] == "none:0.0":
                    continue
                print(
                    f"  {row['condition']:<14}{float(row['caution']) - intact:+10.3f}",
                )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "signal", "condition", "caution", "sd", "batches"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {arguments.out}")
    print(
        "\nRead the swing, and read the sd next to it. A signal whose "
        "between-condition\nswing is smaller than its within-condition spread "
        "cannot be acted on per tick,\nwhatever its mean does.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
