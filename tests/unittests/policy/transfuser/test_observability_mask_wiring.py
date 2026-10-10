"""The observability mask must reach the model, and must replace the bias.

``use_residual_gain`` was read by exactly one backbone for months while every
other silently ignored it, so a run that set it trained the control and the
null result was blamed on the idea. These tests exist so the same thing cannot
happen to the mask.

Four properties, and the third is the one that carries the experiment:

* setting the flag builds the modules, clearing it builds none;
* a masked model starts where the unmasked one starts, so any difference the
  closed loop reports is learned rather than an initialisation artefact;
* a masked block does not also bias the modality logits, because a block doing
  both would leave neither intervention attributable;
* the mask refuses to build without the gate head whose logits it consumes.
"""

import pytest
import torch

from lead.config import LeadConfig
from lead.policy.transfuser.encoder.backbone_deformable_fusion import (
    DeformableFusionBackbone,
)
from lead.policy.transfuser.encoder.observability_mask import ObservabilityMask

_DEFORMABLE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion"
    ":DeformableFusionBackbone"
)


def _config(masked: bool, gated: bool = True) -> LeadConfig:
    """Build a config for a deformable backbone with the flags set.

    Args:
        masked: Value for ``use_observability_mask``.
        gated: Value for ``use_observability_gate``.

    Returns:
        The configured tree.
    """
    config = LeadConfig()
    transfuser = config.policy.transfuser
    transfuser.backbone_target = _DEFORMABLE
    transfuser.deformable_calibrated_reference = True
    transfuser.use_observability = True
    transfuser.use_observability_gate = gated
    transfuser.use_observability_mask = masked
    return config


def _masks(backbone: torch.nn.Module) -> list[ObservabilityMask]:
    """Every mask module the backbone built.

    Args:
        backbone: The backbone to search.

    Returns:
        The mask modules, in traversal order.
    """
    return [m for m in backbone.modules() if isinstance(m, ObservabilityMask)]


@pytest.mark.parametrize("masked", [True, False])
def test_flag_builds_the_modules(masked: bool) -> None:
    """The flag decides whether any mask exists at all.

    Args:
        masked: Whether the flag is set.
    """
    backbone = DeformableFusionBackbone(_config(masked))
    built = _masks(backbone)
    if masked:
        assert built, "use_observability_mask was set and no mask was built"
        assert len(built) == backbone.config.n_layer * 4, (
            f"expected one mask per block of every stage, got {len(built)}"
        )
    else:
        assert not built, "a mask was built with the flag clear"


def test_masked_model_starts_where_the_control_starts() -> None:
    """At initialisation the mask admits almost none of its prior."""
    torch.manual_seed(0)
    masked = DeformableFusionBackbone(_config(True)).eval()
    torch.manual_seed(0)
    control = DeformableFusionBackbone(_config(False)).eval()

    image = torch.randn(1, 3, 384, 1152)
    lidar = torch.randn(1, 1, 320, 384)
    with torch.no_grad():
        masked_lidar, masked_image = masked._forward(image, lidar)  # noqa: SLF001
        control_lidar, control_image = control._forward(image, lidar)  # noqa: SLF001

    assert (masked_lidar - control_lidar).abs().max() < 0.1
    assert (masked_image - control_image).abs().max() < 1.0


def test_mask_replaces_the_bias_rather_than_joining_it() -> None:
    """A masked block hands the operator no modality bias.

    The two are alternative uses of one signal. A block that applied both would
    make the closed-loop difference unattributable to either.
    """
    backbone = DeformableFusionBackbone(_config(True))
    block = backbone.transformers[0].blocks.blocks[0]
    assert block.gate is not None, "the mask needs the gate head for its signal"
    assert block.observability_mask is not None

    seen: list[torch.Tensor | None] = []
    original = block.attn.forward

    def record(tokens, bias, *args, **kwargs):  # noqa: ANN001, ANN202
        seen.append(bias)
        return original(tokens, bias, *args, **kwargs)

    block.attn.forward = record  # type: ignore[method-assign]
    tokens = 12 * 36 + 10 * 12
    block(torch.randn(1, tokens, backbone.transformers[0].n_embd))
    assert seen == [None], f"masked block still passed a modality bias: {seen}"


def test_mask_without_gate_is_refused() -> None:
    """Building a mask with no head to read is a configuration error."""
    with pytest.raises(ValueError, match="use_observability_mask"):
        DeformableFusionBackbone(_config(True, gated=False))
