# -*- coding: utf-8 -*-
"""Prove use_observability_mask both builds the mask and runs it.

Two flags in this repository have been read by one code path and silently
ignored by another. use_residual_gain was one, and the guard in
Transfuser.__init__ exists because of it. A flag that builds a module the
forward pass never calls costs a full run to discover: the model trains, the
loss falls, the evaluation scores, and the row is a duplicate of its own
control.

So this checks the two things separately, because passing one does not imply
the other.

  1. Construction differs. Off gives zero masks, on gives some. If both give
     the same count the flag is inert and whatever built them is something
     else -- use_observability, most likely.

  2. The forward pass reaches them. A hook on every mask records whether it
     was called while the backbone ran on synthetic tensors of the real input
     shapes. A module that is built and never called is the exact failure the
     first check cannot see.

Runs on the CPU and touches nothing the trainer is using.
"""

import sys

import torch

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.observability_mask import ObservabilityMask
from lead.policy.transfuser.transfuser import Transfuser

DEFORMABLE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion"
    ":DeformableFusionBackbone"
)


def build(mask: bool) -> Transfuser:
    """Build the policy exactly as the run script configures it.

    Args:
        mask: Value of the flag under test; everything else is the run's.

    Returns:
        The constructed policy.
    """
    config = LeadConfig()
    transfuser = config.policy.transfuser
    transfuser.backbone_target = DEFORMABLE
    transfuser.deformable_calibrated_reference = True
    transfuser.use_observability = True
    transfuser.use_observability_gate = True
    transfuser.use_observability_mask = mask
    transfuser.use_planning_decoder = True
    config.training.data.use_sensor_degradation = True
    return Transfuser(config)


def masks_of(model: Transfuser) -> list[ObservabilityMask]:
    """Every mask module the policy built.

    Args:
        model: The policy.

    Returns:
        The mask modules, in module order.
    """
    return [m for m in model.modules() if isinstance(m, ObservabilityMask)]


print("=== check 1: does the flag change what is built? ===")
off = len(masks_of(build(mask=False)))
model = build(mask=True)
on_masks = masks_of(model)
print(f"  flag off : {off} mask module(s)")
print(f"  flag on  : {len(on_masks)} mask module(s)")
if off != 0:
    sys.exit(f"FATAL: {off} masks built with the flag OFF; the flag is not what "
             f"builds them, so the run would not be a one-flag comparison.")
if not on_masks:
    sys.exit("FATAL: no masks built with the flag ON.")
print("  OK: construction is controlled by the flag")

print()
print("=== check 2: does the forward pass actually call them? ===")
called: list[str] = []
for index, module in enumerate(on_masks):
    module.register_forward_hook(
        lambda _m, _i, _o, name=f"mask[{index}]": called.append(name),
    )

transfuser = model.lead_config.policy.transfuser
image = torch.zeros(
    1, 3, transfuser.final_image_height, transfuser.final_image_width,
)
lidar = torch.zeros(
    1, 1, transfuser.lidar_height_pixel, transfuser.lidar_width_pixel,
)
model.eval()
with torch.no_grad():
    model.backbone._forward(image, lidar)  # noqa: SLF001

print(f"  masks called during one backbone forward: {len(called)} of {len(on_masks)}")
if len(called) != len(on_masks):
    sys.exit("FATAL: a mask was built but never called. The flag would train a "
             "model identical to its control and blame the idea.")
print("  OK: every mask runs")

print()
print("VERDICT: use_observability_mask builds the masks and the forward reaches "
      "every one of them.")
