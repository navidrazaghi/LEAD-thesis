"""Where the retrained baseline's plans differ from the expert's, and from rung2a's.

Two explanations for its timeouts have already been measured and refused. It is
not the inertia problem: it picks the stop class 26.7% of the time against the
old baseline's 36.0%, so longer training reduced stopping rather than entrenching
it. And it is not a speed ceiling: the expert's own labels put 79% of frames in
the 8 m/s class and never use the five classes above it, so the model's ceiling
is the data's ceiling and not a defect.

What is left is where the car goes rather than how fast. rung2a has almost
identical speed statistics and times out on nothing, so the difference should be
visible in the plan itself. This measures it on cached frames, against the same
labels the training loss uses:

``route``
    The spatial path the policy predicts, scored as average and final
    displacement error against the expert's route. This is the quantity the
    controller steers along, so an error here is a car going somewhere else.

``future waypoints``
    The temporal plan, scored the same way. Route error and waypoint error can
    come apart -- a plan can follow the right path at the wrong times -- and
    which one moves says which half of the planner is at fault.

Frames, order and batch size are held fixed across models so the numbers are
comparable rather than merely present.
"""

import argparse
import pathlib
import statistics
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402


def _displacement(predicted, label):
    """Average and final displacement error between two point sequences.

    Args:
        predicted: ``(batch, steps, 2)``.
        label: The same shape.

    Returns:
        ``(ade, fde)`` as Python floats, in metres.
    """
    predicted = predicted.float()
    label = label.float()
    steps = min(predicted.shape[1], label.shape[1])
    distance = (predicted[:, :steps, :] - label[:, :steps, :]).norm(dim=-1)
    return distance.mean().item(), distance[:, -1].mean().item()


def measure(model, lead_config, loader, batches, device):
    """Route and waypoint displacement error over a fixed set of frames.

    Args:
        model: The loaded policy.
        lead_config: Its config.
        loader: Loader over the probe frames.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        ``{name: (ade, fde)}`` for whichever of the two plans this model emits.
    """
    gathered: dict[str, list[tuple[float, float]]] = {}
    optimization = lead_config.training.optimization
    model.eval()
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            if seen >= batches:
                break
            batch = to_device(batch, device)
            with autocast(
                device_type=device.type,
                dtype=optimization.torch_dtype,
                enabled=optimization.use_mixed_precision_training,
            ):
                out = model(batch)
            for name, attribute, key in (
                ("route", "route", "route"),
                ("waypoints", "future_waypoints", "future_waypoints"),
            ):
                predicted = getattr(out, attribute, None)
                label = batch.get(key)
                if predicted is None or label is None:
                    continue
                gathered.setdefault(name, []).append(
                    _displacement(predicted, label),
                )
            seen += 1
    return {
        name: (
            statistics.mean(a for a, _ in values),
            statistics.mean(f for _, f in values),
        )
        for name, values in gathered.items()
    }


def main() -> int:
    """Compare planning error across checkpoints.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
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
        rows.append(
            (name, measure(model, lead_config, loader, arguments.batches, device)),
        )

    print(
        f"\n  {arguments.batches} batches of {arguments.batch_size}, "
        f"identical frames and order. Errors in metres.\n",
    )
    header = (
        f"  {'model':<28}{'route ADE':>11}{'route FDE':>11}"
        f"{'wp ADE':>10}{'wp FDE':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, results in rows:
        route = results.get("route")
        way = results.get("waypoints")
        print(
            f"  {name:<28}"
            f"{(f'{route[0]:.3f}' if route else '-'):>11}"
            f"{(f'{route[1]:.3f}' if route else '-'):>11}"
            f"{(f'{way[0]:.3f}' if way else '-'):>10}"
            f"{(f'{way[1]:.3f}' if way else '-'):>10}",
        )
    print(
        "\n  Route error is what the controller steers along, so a model that "
        "plans the right\n  speed and the wrong path drives confidently to the "
        "wrong place.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
