"""The residual gain has to be built, not merely configured.

`use_residual_gain` was read by exactly one backbone for months. Setting it on
any other one was accepted, written into the run's config.yaml, and ignored --
which would have trained a rung indistinguishable from its control and produced
a null result about nothing. These tests exist so that cannot recur silently:
one checks the flag reaches the parameters, one checks that a backbone dropping
it raises rather than shrugging, and one checks a gained model starts exactly
where its control does.

They are the same shape as tests/unittests/config/test_overridable_knobs.py,
which guards the other member of this family -- a config value that looks
settable and is not.
"""

import pytest
import torch
from torch import nn

from lead.api.abstract_policy import build_policy
from lead.config import load_lead_config
from lead.policy.transfuser.encoder import transfuser_backbone
from lead.policy.transfuser.encoder.residual_gain import ResidualGain
from lead.policy.transfuser.encoder.transfuser_backbone import Block

_DENSE = "lead.policy.transfuser.encoder.transfuser_backbone:TransfuserBackbone"
_DEFORMABLE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion:"
    "DeformableFusionBackbone"
)


def _policy(backbone_target: str, gained: bool):
    """Build a policy with one backbone and one setting of the gain.

    Args:
        backbone_target: The backbone to build.
        gained: Whether to ask for the residual gain.

    Returns:
        The built policy.
    """
    config = load_lead_config()
    config.policy.transfuser.backbone_target = backbone_target
    config.policy.transfuser.use_residual_gain = gained
    return build_policy(config)


def _count_gains(module: nn.Module) -> int:
    """How many residual gains a module tree actually contains.

    Args:
        module: The tree to search.

    Returns:
        The number of :class:`ResidualGain` instances.
    """
    return sum(isinstance(child, ResidualGain) for child in module.modules())


@pytest.mark.parametrize("backbone_target", [_DENSE, _DEFORMABLE])
def test_flag_reaches_the_parameters(backbone_target: str) -> None:
    """Turning the flag on must add gains, and off must add none.

    Args:
        backbone_target: The backbone under test.
    """
    off = _policy(backbone_target, gained=False)
    on = _policy(backbone_target, gained=True)

    assert _count_gains(off) == 0
    assert _count_gains(on) > 0, (
        f"{backbone_target} accepted use_residual_gain and built no gain, so "
        f"the flag would have been silently ignored."
    )
    assert sum(p.numel() for p in on.parameters()) > sum(
        p.numel() for p in off.parameters()
    )


def test_a_backbone_that_drops_the_flag_raises(monkeypatch) -> None:
    """The guard has to fire, or it is only decorative.

    A guard that never rejects anything looks identical to a guard that works.
    A backbone which ignores the flag is simulated by making the block discard
    it, which is exactly what "does not pass it through" means from outside.

    Args:
        monkeypatch: pytest's attribute patcher.
    """
    original = transfuser_backbone.Block

    class BlockIgnoringTheFlag(original):
        """A block that accepts the argument and does nothing with it."""

        def __init__(self, *args, **kwargs) -> None:
            """Drop the gain, keeping every other argument.

            Args:
                *args: Positional arguments, with any gain flag removed.
                **kwargs: Keyword arguments, with any gain flag removed.
            """
            kwargs.pop("gained", None)
            super().__init__(*args[:5], **kwargs)

    monkeypatch.setattr(transfuser_backbone, "Block", BlockIgnoringTheFlag)
    with pytest.raises(TypeError, match="use_residual_gain"):
        _policy(_DENSE, gained=True)


def test_a_gained_block_starts_as_an_ungained_one() -> None:
    """The gain must be exactly one at initialisation.

    The zeroed projection is what makes the comparison against the control fair
    at step zero. A generic weight initialisation running after it would break
    that with no error anywhere, which is why the backbone re-zeroes it and why
    this checks the consequence rather than the call.
    """
    torch.manual_seed(0)
    plain = Block(32, 4, 4, 0.0, 0.0, False)
    torch.manual_seed(0)
    gained = Block(32, 4, 4, 0.0, 0.0, True)
    gained.load_state_dict(plain.state_dict(), strict=False)
    gained.residual_gain.reset_parameters()

    x = torch.randn(2, 7, 32)
    plain.eval()
    gained.eval()
    with torch.no_grad():
        assert torch.allclose(plain(x), gained(x), atol=1e-6)
        assert torch.allclose(
            gained.residual_gain(torch.randn(2, 7, 32)),
            torch.ones(2, 7, 1),
            atol=1e-6,
        )
