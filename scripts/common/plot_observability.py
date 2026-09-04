"""Draw one frame under clean and damaged sensors, for the thesis figure.

The tables say the gate reallocates attention. They do not show *where*, and a
reader has no reason to take the mechanism on faith from a column of numbers.
This puts the same frame through the model three times -- intact, camera
destroyed, LiDAR destroyed -- and lays the inputs beside what the model believes
about them.

Each row is one sensor condition. The columns:

1. the stitched camera image the model actually received, damage included;
2. the LiDAR BEV raster it actually received;
3. and 4. the observability head's belief about what each modality resolves,
   per BEV cell;
5. the gate's camera-minus-LiDAR preference over the BEV tokens, on a diverging
   scale centred at zero -- red where the query leans on the camera, blue where
   it leans on the LiDAR.

Column 5 is the figure's point. Between the clean row and the damaged rows it
should swing bodily from one end of the scale to the other, and it should do so
in the places the damaged modality was carrying.

Every panel is drawn with the ego forward axis pointing up and the ego's right
hand to the right, which is not how the arrays are stored: the BEV grids are
indexed ``[y, x]`` with y running left-to-right and x running back-to-front, so
each one is transposed and flipped on the way to the page.

Usage::

    python scripts/common/plot_observability.py \\
      --model outputs/rung3_observability_gated_post \\
      --out results/observability_figure.png
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.amp.autocast_mode import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

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

_DAMAGE_SEED = 20260820
_CONDITIONS = (("none", 0.0, "intact"), ("camera", 1.0, "camera destroyed"),
               ("lidar", 1.0, "LiDAR destroyed"))


def ego_up(grid: np.ndarray) -> np.ndarray:
    """Reorient a ``[y, x]`` BEV grid so the ego drives up the page.

    Args:
        grid: A BEV grid indexed ``[y, x]``.

    Returns:
        The grid with forward up and the ego's right to the right.
    """
    return grid.T[::-1]


class GateProbe:
    """Collects the gate logits of every block, keyed by nothing but order."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Hook every gated block.

        Args:
            model: The loaded policy.
        """
        self.logits: list[torch.Tensor] = []
        self._handles = [
            block.gate.register_forward_hook(self._hook)
            for block in model.modules()
            if isinstance(block, DeformableBlock) and block.gate is not None
        ]

    def _hook(self, _module, _inputs, output) -> None:
        """Record one block's gate logits.

        Args:
            _module: Unused.
            _inputs: Unused.
            output: The gate logits.
        """
        self.logits.append(output.detach().float())

    def close(self) -> None:
        """Remove every hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def preference(self, sample: int, rows: int, cols: int) -> np.ndarray | None:
        """The gate's camera-minus-LiDAR preference over the BEV tokens.

        Args:
            sample: Which sample of the batch to read.
            rows: BEV token grid rows.
            cols: BEV token grid columns.

        Returns:
            The preference map, or None when the model has no gate.
        """
        if not self.logits:
            return None
        # Image tokens come first and have no BEV footprint, so the BEV grid is
        # the tail of the sequence.
        maps = [
            (logits[sample, -rows * cols :, 0] - logits[sample, -rows * cols :, 1])
            .reshape(rows, cols)
            .cpu()
            .numpy()
            for logits in self.logits
        ]
        return np.mean(maps, axis=0)


def main() -> None:
    """Render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=ROOT / "outputs/rung3_observability_gated_post",
    )
    parser.add_argument("--batch-index", type=int, default=3)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "results/observability_figure.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    with (args.model / "config.yaml").open(encoding="utf-8") as handle:
        stored = yaml.safe_load(handle)
    stored.pop("evaluation", None)
    lead_config = load_lead_config(loaded_config=stored, raise_on_unknown_key=False)
    policy = PolicyRunner(
        lead_config=lead_config,
        model_path=str(args.model),
        device=device,
    ).policy
    policy.eval()
    config = lead_config.policy.transfuser

    dataset = policy.build_dataset()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        num_workers=2,
    )
    source = None
    for index, batch in enumerate(loader):
        if index == args.batch_index:
            source = batch
            break
    if source is None:
        raise SystemExit(f"batch {args.batch_index} is past the end of the dataset")

    panels = []
    for modality, severity, label in _CONDITIONS:
        batch = {
            key: (value.clone() if isinstance(value, torch.Tensor) else value)
            for key, value in source.items()
        }
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        generator = torch.Generator(device=device)
        generator.manual_seed(_DAMAGE_SEED)
        batch = degrade_batch(batch, modality, severity, generator)

        probe = GateProbe(policy)
        try:
            with torch.inference_mode(), autocast(
                device_type="cuda",
                dtype=lead_config.training.optimization.torch_dtype,
                enabled=lead_config.training.optimization.use_mixed_precision_training,
            ):
                prediction = policy(batch)
            preference = probe.preference(
                args.sample_index,
                config.lidar_bev_grid_rows,
                config.lidar_bev_grid_cols,
            )
        finally:
            probe.close()

        sample = args.sample_index
        rgb = (
            batch["rgb"][sample].float().clamp(0, 255).div(255).permute(1, 2, 0).cpu().numpy()
        )
        lidar = batch["rasterized_lidar"][sample].float().sum(0).cpu().numpy()
        observability = getattr(prediction, "observability", None)
        if observability is not None:
            observability = torch.sigmoid(observability[sample].float()).cpu().numpy()
        panels.append((label, rgb, lidar, observability, preference))

    columns = [
        "camera input",
        "LiDAR input",
        "obs: camera",
        "obs: LiDAR",
        "gate: red leans camera, blue leans LiDAR",
    ]
    # The stitched camera panel is three cameras wide and needs the room, or it
    # shrinks to a strip and drags its column title out of line with the rest.
    figure, axes = plt.subplots(
        len(panels),
        len(columns),
        figsize=(19.0, 3.1 * len(panels)),
        gridspec_kw={"width_ratios": [2.9, 1.0, 1.0, 1.0, 1.0]},
    )
    limit = max(
        (np.abs(p[4]).max() for p in panels if p[4] is not None),
        default=1.0,
    )

    gate_image = observability_image = None
    for row, (label, rgb, lidar, observability, preference) in enumerate(panels):
        axes[row][0].imshow(rgb)
        axes[row][0].set_ylabel(label, fontsize=12, labelpad=10)
        axes[row][1].imshow(ego_up(lidar), cmap="bone")
        for offset, channel in enumerate(ObservabilityChannel):
            axis = axes[row][2 + offset]
            if observability is None:
                axis.text(0.5, 0.5, "no head", ha="center", va="center")
            else:
                observability_image = axis.imshow(
                    ego_up(observability[channel]),
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
        axis = axes[row][4]
        if preference is None:
            axis.text(0.5, 0.5, "no gate", ha="center", va="center")
        else:
            gate_image = axis.imshow(
                ego_up(preference),
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
        for column in range(len(columns)):
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
            # Pin every panel to the top of its cell so the column titles sit on
            # one line despite the wildly different aspect ratios.
            axes[row][column].set_anchor("N")
            if row == 0:
                axes[row][column].set_title(columns[column], fontsize=12, pad=8)

    figure.suptitle(
        f"{args.model.name}: what the model receives, and what it believes about it",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))

    # One colourbar per quantity rather than one per row: the scales are shared
    # down each column, so repeating them three times says nothing extra.
    if observability_image is not None:
        figure.colorbar(
            observability_image,
            ax=axes[:, 2:4].ravel().tolist(),
            orientation="horizontal",
            fraction=0.04,
            pad=0.04,
            label="predicted observability",
        )
    if gate_image is not None:
        figure.colorbar(
            gate_image,
            ax=axes[:, 4].ravel().tolist(),
            orientation="horizontal",
            fraction=0.04,
            pad=0.04,
            label="camera - LiDAR gate logit",
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
