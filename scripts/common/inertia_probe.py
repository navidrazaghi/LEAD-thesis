"""How often the policy decides to stop, against how often the expert does.

The retrained baseline times out on 36% of routes against 3% for the run it
replaced, while its perception loss is 5.4x lower. Routes the old model finished
in two minutes the new one drives for forty-five without finishing. That is the
signature of the inertia problem: a policy trained by imitation learns the
spurious correlation between being stopped and staying stopped, because standing
at a red light is over-represented in any driving log.

CILRS's remedy -- predicting speed as an auxiliary task -- is already in this
stack, so the interesting question is not whether to add it but whether it is
working. This measures that directly. Target speed here is a classification over
eight classes whose first is 0 m/s, so "decides to stop" is a class the model
either picks or does not, and the expert's own label is available on the same
frame.

The comparison that matters is between checkpoints rather than against any
absolute number. If a longer-trained model picks the stop class more often than
a shorter-trained one on identical frames, the extra epochs entrenched the
correlation rather than the driving, which is the mechanism this is testing for.
Frames are held fixed and taken in the same order for every model.
"""

import argparse
import collections
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402


def stop_rate(model, lead_config, loader, batches, device):
    """How often the model picks each target-speed class, and what the expert did.

    Args:
        model: The loaded policy.
        lead_config: Its config.
        loader: Loader over the probe frames.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        ``(predicted counts, expert counts, total frames)`` as Counters over
        class indices.

    Raises:
        SystemExit: If the model produces no target-speed distribution, which
            means this checkpoint cannot be probed this way rather than that it
            never stops.
    """
    predicted: collections.Counter = collections.Counter()
    expert: collections.Counter = collections.Counter()
    total = 0
    model.eval()
    seen = 0
    optimization = lead_config.training.optimization
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
            distribution = getattr(out, "target_speed_distribution", None)
            if distribution is None:
                raise SystemExit(
                    "this checkpoint returns no target-speed distribution; it "
                    "has no planning decoder and cannot be probed for inertia.",
                )
            for index in distribution.argmax(dim=-1).tolist():
                predicted[index] += 1
            # The expert's label is a speed in m/s, not a class index -- it is
            # turned into a soft two-hot target by the decoder's loss. Counting
            # int(speed) would bucket 13.9 m/s as class 13, which does not
            # exist. Each value is mapped to its nearest class instead.
            label = batch.get("target_speed")
            if label is not None:
                edges = torch.tensor(
                    lead_config.policy.transfuser.target_speed_classes,
                    device=label.device,
                    dtype=torch.float32,
                )
                nearest = (label.reshape(-1, 1).float() - edges).abs().argmin(dim=-1)
                for index in nearest.tolist():
                    expert[int(index)] += 1
            total += distribution.shape[0]
            seen += 1
    return predicted, expert, total


def main() -> int:
    """Compare the stop rate across checkpoints.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    rows = []
    classes = None
    for pair in arguments.models:
        name, _, path = pair.partition("=")
        lead_config, model = load_model(pathlib.Path(path), device)
        classes = lead_config.policy.transfuser.target_speed_classes
        dataset = model.build_dataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=arguments.workers,
        )
        predicted, expert, total = stop_rate(
            model, lead_config, loader, arguments.batches, device,
        )
        rows.append((name, predicted, expert, total))

    print(f"\n  target-speed classes (m/s): {classes}")
    print(f"  {arguments.batches} batches of {arguments.batch_size}, identical frames\n")

    header = f"  {'model':<28}{'stop rate':>11}{'expert':>9}{'ratio':>8}  distribution"
    print(header)
    print("  " + "-" * (len(header) + 12))
    for name, predicted, expert, total in rows:
        stop = predicted[0] / total if total else 0.0
        expert_stop = expert[0] / total if total and expert else float("nan")
        ratio = stop / expert_stop if expert_stop else float("nan")
        spread = " ".join(
            f"{predicted[i] / total:.2f}" if total else "-"
            for i in range(len(classes or []))
        )
        print(
            f"  {name:<28}{100 * stop:>10.1f}%{100 * expert_stop:>8.1f}%"
            f"{ratio:>8.2f}  {spread}",
        )
    # The expert is the same on every model's frames, so it is printed once and
    # is the row that says whether the high classes are used at all.
    if rows:
        _, _, expert, total = rows[0]
        spread = " ".join(
            f"{expert[i] / total:.2f}" if total else "-"
            for i in range(len(classes or []))
        )
        print(f"  {'expert (the label itself)':<28}{'':>10}{'':>8}{'':>8}  {spread}")

    print(
        "\n  A ratio above one means the model chooses to stop more often than "
        "the expert did\n  on the same frames, which is the inertia problem "
        "measured rather than inferred.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
