"""The oracle mode of the modality gate.

Two things have to hold. Off, the gate must produce exactly what it produced
before the flag existed -- this code is edited while an evaluation chain is
queued behind it, and a drift here would move numbers nobody would think to
re-check. On, the bias must land on the damaged modality and only on it;
reading the modality axis the wrong way round would mask the intact sensor and
still produce a plausible-looking table.
"""

import pytest
import torch

from lead.config import load_lead_config
from lead.policy.transfuser.encoder.observability_gate import (
    _MODALITY_INDEX,
    ObservabilityGate,
)

CAMERA, LIDAR = _MODALITY_INDEX["camera"], _MODALITY_INDEX["lidar"]


@pytest.fixture
def config():
    """A default config tree, with nothing damaged."""
    return load_lead_config()


@pytest.fixture
def gate(config):
    """A gate with a non-zero projection, so "off" is not trivially zero too."""
    gate = ObservabilityGate(n_embd=8, num_levels=2, lead_config=config)
    torch.manual_seed(0)
    torch.nn.init.normal_(gate.head.weight, std=0.5)
    torch.nn.init.normal_(gate.head.bias, std=0.5)
    return gate


@pytest.fixture
def tokens():
    """One small batch of fusion tokens."""
    torch.manual_seed(1)
    return torch.randn(2, 5, 8)


def test_off_is_the_learned_head(gate, tokens):
    """With the flag off the gate is its projection, to the last bit."""
    assert torch.equal(gate(tokens), gate.head(tokens))


def test_off_is_the_default(config):
    """Nothing has to be set to keep the old behaviour."""
    assert config.evaluation.inference.oracle_gate is False


def test_undamaged_oracle_is_neutral(gate, tokens, config):
    """Told nothing is damaged, the oracle asks for no shift at all."""
    config.evaluation.inference.oracle_gate = True
    assert torch.equal(gate(tokens), torch.zeros_like(gate.head(tokens)))


@pytest.mark.parametrize(
    ("modality", "damaged", "intact"),
    [("camera", CAMERA, LIDAR), ("lidar", LIDAR, CAMERA)],
)
def test_oracle_masks_the_damaged_modality(
    gate,
    tokens,
    config,
    modality,
    damaged,
    intact,
):
    """The bias lands on the destroyed modality, and the other is untouched."""
    inference = config.evaluation.inference
    inference.oracle_gate = True
    inference.degrade_modality = modality
    inference.degrade_severity = 1.0

    logits = gate(tokens)
    assert torch.all(logits[..., damaged] == -inference.oracle_gate_strength)
    assert torch.all(logits[..., intact] == 0.0)


def test_severity_scales_the_bias(gate, tokens, config):
    """Half the damage is half the bias, so partial severities stay meaningful."""
    inference = config.evaluation.inference
    inference.oracle_gate = True
    inference.degrade_modality = "camera"
    inference.degrade_severity = 0.5

    logits = gate(tokens)
    assert torch.all(
        logits[..., CAMERA] == -0.5 * inference.oracle_gate_strength,
    )


def test_oracle_refuses_a_spatial_family(gate, tokens, config):
    """A family's severity varies over the image; no scalar stands in for it."""
    inference = config.evaluation.inference
    inference.oracle_gate = True
    inference.degrade_family = "occlusion"

    with pytest.raises(ValueError, match="no ground truth"):
        gate(tokens)


def test_the_mask_is_effectively_a_mask(gate, tokens, config):
    """After the softmax the damaged modality keeps a negligible share.

    The operator normalizes over (modality, point) with one softmax, so the
    check that matters is what the bias leaves behind there, not its size.
    """
    inference = config.evaluation.inference
    inference.oracle_gate = True
    inference.degrade_modality = "lidar"
    inference.degrade_severity = 1.0

    share = torch.softmax(gate(tokens), dim=-1)[..., LIDAR]
    assert torch.all(share < 1e-4)
