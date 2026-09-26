"""Does the ensemble's disagreement move enough to drive a governor?

The observability signal was measured this way before any closed-loop night was
spent on it, and the answer was that it barely moves: about two hundredths
between intact sensors and a destroyed camera, because with one modality gone
the other still resolves the scene and the estimate correctly says so. Useful to
know beforehand rather than after twenty-two hours of simulator.

This asks the same question of the ensemble. The failure mode here is different
and quieter. Members fitted to one dataset converge on each other, so a
collapsed ensemble reports a spread near zero on every frame, under every
condition, and looks exactly like a model that is confident. Two numbers
separate that from a working signal: the spread under intact sensors, which
should be small but not zero, and how far it moves under damage.

An untrained ensemble is worth measuring too, as the ceiling. Its members differ
by initialisation alone and have not yet been pulled toward the same labels, so
if the trained spread is far below it in every condition, they have converged.

That measurement was taken before the fine-tune finished, on four frames, which
makes it a smoke test rather than a number to quote. It reads: about 1.33 m
under intact sensors, 1.32 m with the camera destroyed, 1.42 m with both
destroyed. So an untrained ensemble disagrees a great deal and barely responds
to damage, which is what members whose plans hardly depend on the input would
do.

Writing down what that implies, before the trained numbers exist, so reading
them afterwards is not a matter of taste. Two outcomes are failures and they
look nothing alike. A trained spread still near 1.3 m everywhere means the
members never learned to use the features. A trained spread near zero
everywhere means they converged onto each other and the signal is gone. What
would make this usable is a small spread under intact sensors -- the features
determine the plan and the members agree -- together with a swing under damage
that is a real fraction of the range between those two.
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

from lead.evaluation.inference.caution import ensemble_caution  # noqa: E402
from lead.policy.transfuser.decoder.waypoint_ensemble import (  # noqa: E402
    ensemble_spread,
)
from lead.policy.transfuser.utils.sensor_degradation import (  # noqa: E402
    degrade_batch,
)

# Fixed, so two checkpoints meet identical damage on identical frames.
_DAMAGE_SEED = 20260826

# Every condition the governor could be evaluated under, including the joint one
# the observability signal is blind to by construction.
_CONDITIONS = (
    ("none", 0.0),
    ("camera", 0.5),
    ("camera", 1.0),
    ("lidar", 0.5),
    ("lidar", 1.0),
    ("both", 0.5),
    ("both", 1.0),
)


def spread_under(model, lead_config, loader, condition, batches, device):
    """Per-frame ensemble spread under one condition, in meters.

    Args:
        model: Loaded policy carrying a waypoint ensemble.
        lead_config: The config it was trained with.
        loader: Loader over the probe frames.
        condition: ``(modality, severity)``.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        The per-frame spreads, and the caution each batch mapped to.

    Raises:
        SystemExit: If the model produces no ensemble, which means it was built
            without one rather than that the ensemble had nothing to say.
    """
    modality, severity = condition
    generator = torch.Generator(device=device)
    generator.manual_seed(_DAMAGE_SEED)
    spreads: list[float] = []
    cautions: list[float] = []
    seen = 0
    # Evaluation mode throughout: the decoder layers carry dropout, and in
    # training mode the spread would include a different mask per member rather
    # than a different opinion.
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
            members = getattr(prediction, "waypoint_ensemble", None)
            if members is None:
                raise SystemExit(
                    "the model produced no waypoint ensemble; it was built "
                    "without one, so there is nothing to measure here.",
                )
            spreads.extend(ensemble_spread(members).cpu().tolist())
            cautions.append(ensemble_caution(members, lead_config))
            seen += 1
    return spreads, cautions


def main() -> int:
    """Measure the spread under every condition and report the swing.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=checkpoint-directory pairs.",
    )
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results" / "ensemble_spread.csv",
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

        print(f"\n{name}")
        print(f"  {'condition':<14}{'spread (m)':>12}{'caution':>10}{'frames':>9}")
        intact = None
        for modality, severity in _CONDITIONS:
            spreads, cautions = spread_under(
                model,
                lead_config,
                loader,
                (modality, severity),
                arguments.batches,
                device,
            )
            if not spreads:
                continue
            spread = statistics.mean(spreads)
            caution = statistics.mean(cautions)
            label = f"{modality}:{severity}"
            print(f"  {label:<14}{spread:12.4f}{caution:10.3f}{len(spreads):9d}")
            rows.append(
                {
                    "model": name,
                    "condition": label,
                    "spread_m": f"{spread:.6f}",
                    "caution": f"{caution:.6f}",
                    "frames": len(spreads),
                },
            )
            if modality == "none":
                intact = spread

        if intact is not None:
            print(f"  {'--- swing from intact ---':<14}")
            for row in rows:
                if row["model"] != name or row["condition"] == "none:0.0":
                    continue
                moved = float(row["spread_m"]) - intact
                print(f"  {row['condition']:<14}{moved:+12.4f}")

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "condition", "spread_m", "caution", "frames"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {arguments.out}")
    print(
        "\nRead the swing column, not the spread column. A spread that is small "
        "everywhere\nand moves nowhere is a collapsed ensemble reporting "
        "confidence it has not earned.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
