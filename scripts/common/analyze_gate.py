"""Measure the fusion mechanism directly, without driving anything.

Closed-loop driving score over ten routes is a noisy, low-power endpoint: in the
pilot a single model's per-route score ranged from 0 to 100. Resting the whole
argument on it risks a real effect disappearing under that spread.

The claim, though, is about the gate, and the gate can be measured head-on. Feed
the same frames through with one modality damaged and ask whether the fusion
shifts its reading onto the modality that still works. That runs over thousands
of frames in minutes, needs no simulator, and tests the mechanism rather than a
downstream consequence of it.

What each column means:

``camera_share``
    The fraction of the deformable operator's attention mass landing on the
    image grid, recomputed exactly as the operator computes it: the same softmax
    over the (modality, point) axes, with the gate's bias added. Averaged over
    every query, head, block and stage. Its complement is the LiDAR share.
    Available for any deformable model, gated or not.

``gate_pref``
    The gate's own camera-minus-LiDAR logit, before the operator's logits join
    it. This is the gate's opinion in isolation; ``camera_share`` is what the
    fusion did with that opinion. Gated models only.

``obs_camera`` / ``obs_lidar``
    What the observability head predicts each modality resolves, averaged over
    the cells the expert measured. Models carrying the head only.

``waypoint_l2``
    Mean L2 between predicted and recorded future waypoints. A cheap open-loop
    stand-in for driving quality, at far higher n than a closed-loop sweep.

Read the table by difference from the clean row of the same model. The absolute
level of any column is uninteresting; the claim is that damaging the camera
pushes ``camera_share`` down, damaging the LiDAR pushes it up, and that a gated
model moves further than an ungated one.

One caveat, stated because it bounds how far these numbers reach: the dataset
has no held-out split, so these frames were seen during training. That is
acceptable here because the question is whether the mechanism *responds* to
damage, not whether it generalizes -- severity is drawn randomly per sample
during training, so no frame carries a memorized answer for a given severity.
It would not be acceptable for a generalization claim.

Usage::

    python scripts/common/analyze_gate.py \
      --models rung0=outputs/rung0_baseline_post rung3=outputs/rung3_observability_gated_post \
      --conditions none:0 camera:1.0 lidar:0.5 lidar:1.0 \
      --batches 40
"""

import argparse
import csv
import pathlib
import sys

import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lead.config import load_lead_config  # noqa: E402
from lead.evaluation.inference.policy_runner import PolicyRunner  # noqa: E402
from lead.policy.transfuser.dataloader.observability import (  # noqa: E402
    ObservabilityChannel,
)
from lead.policy.transfuser.encoder.backbone_deformable_fusion import (  # noqa: E402
    DeformableBlock,
)
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

# Fixed, so every model meets identical damage on identical frames.
_DAMAGE_SEED = 20260820

_METRICS = ("camera_share", "gate_pref", "obs_camera", "obs_lidar", "waypoint_l2")


class FusionProbe:
    """Captures what each deformable block attended to, without changing it.

    The operator does not return its attention weights, and adding a return
    value would mean editing a model that is mid-training. This hooks the two
    linear layers the weights are computed from and repeats that arithmetic
    here -- same view, same optional gate bias, same softmax over the
    ``(modality, point)`` axes -- so the numbers are the operator's own rather
    than an approximation of them.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Attach to every deformable block of the model.

        Args:
            model: The loaded policy.
        """
        self.blocks: list[DeformableBlock] = [
            module for module in model.modules() if isinstance(module, DeformableBlock)
        ]
        self._weight_logits: dict[int, torch.Tensor] = {}
        self._gate_logits: dict[int, torch.Tensor] = {}
        self._handles: list = []
        for index, block in enumerate(self.blocks):
            self._handles.append(
                block.attn.attention_weights.register_forward_hook(
                    self._store(self._weight_logits, index),
                ),
            )
            if block.gate is not None:
                self._handles.append(
                    block.gate.register_forward_hook(
                        self._store(self._gate_logits, index),
                    ),
                )

    @staticmethod
    def _store(sink: dict, index: int):
        """Build a hook filing one module's output under ``index``.

        Args:
            sink: Dict the output is stored in.
            index: Which block this hook belongs to.

        Returns:
            The hook.
        """

        def hook(_module, _inputs, output):
            sink[index] = output.detach()

        return hook

    def close(self) -> None:
        """Remove every hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def read(self) -> tuple[float | None, float | None]:
        """Summarize the forward pass that just ran.

        Returns:
            Mean camera share of the attention mass and mean gate preference,
            either being None when the model has no such thing.
        """
        if not self.blocks:
            return None, None

        shares: list[float] = []
        preferences: list[float] = []
        for index, block in enumerate(self.blocks):
            logits = self._weight_logits.get(index)
            if logits is None:
                continue
            attention = block.attn
            batch, tokens = logits.shape[0], logits.shape[1]
            logits = logits.float().view(
                batch,
                tokens,
                attention.n_head,
                attention.num_levels,
                attention.num_points,
            )
            gate = self._gate_logits.get(index)
            if gate is not None:
                gate = gate.float()
                logits = logits + gate[:, :, None, :, None]
                preferences.append((gate[..., 0] - gate[..., 1]).mean().item())
            weights = F.softmax(logits.flatten(-2), dim=-1).view(logits.shape)
            # Sum each modality's points, then average the heads: what is left
            # is how the query split its reading between the two grids.
            modality = weights.sum(dim=-1).mean(dim=2)
            shares.append(modality[..., 0].mean().item())

        self._weight_logits.clear()
        self._gate_logits.clear()
        return (
            sum(shares) / len(shares) if shares else None,
            sum(preferences) / len(preferences) if preferences else None,
        )


def to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch's tensors onto the device, leaving everything else alone.

    Args:
        batch: The collated batch.
        device: Target device.

    Returns:
        The batch, tensors moved.
    """
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def load_model(checkpoint: pathlib.Path, device: torch.device):
    """Rebuild a trained policy from its checkpoint directory.

    Args:
        checkpoint: Directory holding ``config.yaml`` and the weights.
        device: Device to load onto.

    Returns:
        The config it was trained with, and the loaded policy.
    """
    with (checkpoint / "config.yaml").open(encoding="utf-8") as handle:
        stored = yaml.safe_load(handle)
    # How it was trained is in the file; how it is probed is decided here.
    stored.pop("evaluation", None)
    lead_config = load_lead_config(loaded_config=stored, raise_on_unknown_key=False)
    runner = PolicyRunner(
        lead_config=lead_config,
        model_path=str(checkpoint),
        device=device,
    )
    return lead_config, runner.policy


def measure(model, lead_config, loader, condition, batches, device) -> dict:
    """Run one model over one condition and summarize what the fusion did.

    Args:
        model: The loaded policy, in eval mode.
        lead_config: The config it was trained with.
        loader: Loader over the probe frames.
        condition: ``(modality, severity)``.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        One row of the results table.
    """
    modality, severity = condition
    probe = FusionProbe(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(_DAMAGE_SEED)

    totals: dict[str, list[float]] = {key: [] for key in _METRICS}
    seen = 0
    try:
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

                share, preference = probe.read()
                if share is not None:
                    totals["camera_share"].append(share)
                if preference is not None:
                    totals["gate_pref"].append(preference)

                observability = getattr(prediction, "observability", None)
                if observability is not None and "observability_mask" in batch:
                    probability = torch.sigmoid(observability.float())
                    mask = batch["observability_mask"].float()
                    for channel in ObservabilityChannel:
                        weight = mask[:, channel].sum().clamp(min=1.0)
                        totals[f"obs_{channel.name.lower()}"].append(
                            (probability[:, channel] * mask[:, channel])
                            .sum()
                            .div(weight)
                            .item(),
                        )

                predicted = getattr(prediction, "future_waypoints", None)
                if predicted is not None and "future_waypoints" in batch:
                    label = batch["future_waypoints"].float()
                    predicted = predicted.float()
                    horizon = min(predicted.shape[1], label.shape[1])
                    totals["waypoint_l2"].append(
                        (predicted[:, :horizon] - label[:, :horizon])
                        .norm(dim=-1)
                        .mean()
                        .item(),
                    )
                seen += 1
    finally:
        probe.close()

    row: dict = {"batches": seen}
    for key, values in totals.items():
        row[key] = sum(values) / len(values) if values else None
    return row


def show(value) -> str:
    """Format a possibly-missing number for the progress line.

    Args:
        value: The number, or None.

    Returns:
        The formatted value.
    """
    return "--" if value is None else f"{value:.4f}"


def main() -> None:
    """Probe every model under every condition and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="name=checkpoint_dir",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["none:0", "camera:1.0", "lidar:0.5", "lidar:1.0"],
    )
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results/mechanism.csv",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    conditions = [pair.split(":", 1) for pair in args.conditions]
    rows = []

    for entry in args.models:
        name, _, path = entry.partition("=")
        print(f"\n=== {name} ===", flush=True)
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()

        dataset = model.build_dataset()
        # No shuffle: every model and every condition must see the same frames
        # in the same order, or the comparison is between samples rather than
        # between models.
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )

        for modality, severity in conditions:
            row = measure(
                model,
                lead_config,
                loader,
                (modality, float(severity)),
                args.batches,
                device,
            )
            row["model"] = name
            row["condition"] = f"{modality}:{severity}"
            rows.append(row)
            print(
                f"  {row['condition']:<12} "
                f"camera_share={show(row['camera_share'])} "
                f"gate_pref={show(row['gate_pref'])} "
                f"wp_l2={show(row['waypoint_l2'])}",
                flush=True,
            )
        del model, loader
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "condition", "batches", *_METRICS]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
