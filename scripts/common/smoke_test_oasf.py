"""End-to-end smoke test of the observability-aware fusion, before any training.

A one-epoch training run is the obvious smoke test, and it is the wrong one for
this change: it would pass while the gate silently never trains. The gate's
logits leave the backbone through a list the block stack fills during forward,
and if Dynamo traces around that mutation the list comes back empty, the loss
term is skipped, and the run still reports success. Everything here is aimed at
faults of that shape — the ones a green training log would hide.

Every other head is turned off so a failure points at this change and nothing
else. Run it before the first real training run, and again after any edit to
the fusion path:

    LEAD_RUNTIME_TYPE_CHECKING=false TIMM_USE_OLD_CACHE=1 \
        python scripts/common/smoke_test_oasf.py --device cuda
"""

from __future__ import annotations

import argparse

import torch

from lead.config import load_lead_config
from lead.policy.transfuser.dataloader.observability import (
    NUM_OBSERVABILITY_CHANNELS,
)
from lead.policy.transfuser.transfuser import Transfuser

BACKBONE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion:DeformableFusionBackbone"
)


def build_config(gated: bool, mixed_precision: bool = False):
    """A config with only the fusion path under test enabled.

    Args:
        gated: Whether to switch the observability gate on.
        mixed_precision: Whether the backbone should cast its inputs to
            bfloat16. Only meaningful inside an autocast region: the trainer
            supplies one, a bare forward pass does not, and calling the model
            outside one with this on hands bf16 inputs to fp32 weights.

    Returns:
        The resolved config tree.
    """
    return load_lead_config(
        {
            "training": {
                "optimization": {"use_mixed_precision_training": mixed_precision},
            },
            "policy": {
                "transfuser": {
                    "backbone_target": BACKBONE,
                    "deformable_calibrated_reference": True,
                    "use_observability": True,
                    "use_observability_gate": gated,
                    # Off so a failure cannot come from anywhere else.
                    "use_semantic": False,
                    "use_depth": False,
                    "use_bev_semantic": False,
                    "detect_boxes": False,
                    "use_radar_detection": False,
                    "use_planning_decoder": False,
                },
            },
        },
    )


def build_batch(config, batch_size: int, device: torch.device) -> dict:
    """A synthetic batch shaped like the collated one the trainer passes.

    Args:
        config: The ``policy.transfuser`` config section.
        batch_size: How many samples to fake.
        device: Where to put the tensors.

    Returns:
        The batch dict.
    """
    cell_rows = config.lidar_height_pixel // config.bev_downsample_factor
    cell_cols = config.lidar_width_pixel // config.bev_downsample_factor
    target = torch.rand(
        batch_size,
        NUM_OBSERVABILITY_CHANNELS,
        cell_rows,
        cell_cols,
        device=device,
    )
    # Half the cells measured, like a scene with a few actors in it.
    mask = (torch.rand_like(target) > 0.5).float()
    return {
        "rgb": torch.randint(
            0,
            255,
            (batch_size, 3, config.final_image_height, config.final_image_width),
            dtype=torch.uint8,
            device=device,
        ),
        "rasterized_lidar": torch.rand(
            batch_size,
            1,
            config.lidar_height_pixel,
            config.lidar_width_pixel,
            device=device,
        ),
        "observability": target,
        "observability_mask": mask,
    }


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print one check's outcome.

    Args:
        label: What was checked.
        ok: Whether it held.
        detail: Extra context to print alongside.

    Returns:
        ``ok``, so callers can accumulate.
    """
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return ok


def run(model, batch, expected_gates: int) -> bool:
    """Forward, loss and backward, checking what a training log would not show.

    Args:
        model: The policy under test.
        batch: The batch to push through.
        expected_gates: How many gated blocks should have reported logits.

    Returns:
        Whether every check passed.
    """
    passed = True
    prediction = model(batch)

    # An ungated model reports None rather than an empty list, and both mean
    # the same thing here: nothing gated ran.
    gates = prediction.observability_gate
    found = 0 if gates is None else len(gates)
    passed &= check(
        "gate logits reach the prediction",
        found == expected_gates,
        f"got {found}, want {expected_gates}",
    )
    if gates:
        tokens = gates[0].shape[1]
        passed &= check(
            "gate logits are shaped (batch, tokens, modalities)",
            gates[0].shape == (batch["rgb"].shape[0], tokens, 2),
            str(tuple(gates[0].shape)),
        )

    passed &= check(
        "observability head produced a map",
        prediction.observability is not None,
        str(tuple(prediction.observability.shape))
        if prediction.observability is not None
        else "",
    )

    losses, _ = model.compute_loss(prediction, batch)
    passed &= check("observability loss present", "loss_observability" in losses)
    if expected_gates:
        passed &= check(
            "gate loss present",
            "loss_observability_gate" in losses,
            "" if "loss_observability_gate" in losses else "SILENTLY SKIPPED",
        )
    for name, value in losses.items():
        passed &= check(
            f"{name} is finite",
            bool(torch.isfinite(value).all()),
            f"{value.item():.4f}",
        )

    total = sum(losses.values())
    total.backward()

    gated_params = [
        (name, p)
        for name, p in model.named_parameters()
        if ".gate." in name or "observability_decoder" in name
    ]
    with_grad = [
        n for n, p in gated_params if p.grad is not None and p.grad.abs().sum() > 0
    ]
    passed &= check(
        "gradient reaches the gate and the observability head",
        len(with_grad) == len(gated_params) and bool(gated_params),
        f"{len(with_grad)}/{len(gated_params)} tensors",
    )
    return passed


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip the torch.compile pass, which is the slow one.",
    )
    return parser.parse_args()


def main() -> None:
    """Run every stage and exit non-zero if any check failed."""
    args = parse_args()
    device = torch.device(args.device)
    passed = True

    print("[1] ungated: the deformable backbone and observability head alone")
    config = build_config(gated=False)
    model = Transfuser(config).to(device)
    batch = build_batch(config.policy.transfuser, args.batch_size, device)
    passed &= run(model, batch, expected_gates=0)

    print("\n[2] gated: the modality gate wired into the fusion")
    config = build_config(gated=True)
    model = Transfuser(config).to(device)
    # One gated block per fusion layer, over the four backbone stages.
    expected = config.policy.transfuser.n_layer * 4
    batch = build_batch(config.policy.transfuser, args.batch_size, device)
    passed &= run(model, batch, expected_gates=expected)

    if device.type == "cuda":
        print("\n[3] gated under bf16 autocast, the precision training uses")
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            passed &= run(model, batch, expected_gates=expected)

    if not args.skip_compile:
        print("\n[4] gated under torch.compile, which training turns on by default")
        print("    (the list the gate logits leave through is what Dynamo may drop)")
        config = build_config(gated=True, mixed_precision=device.type == "cuda")
        model = Transfuser(config).to(device)
        model.forward = torch.compile(model.forward)
        batch = build_batch(config.policy.transfuser, args.batch_size, device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                passed &= run(model, batch, expected_gates=expected)
        else:
            passed &= run(model, batch, expected_gates=expected)

    print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
