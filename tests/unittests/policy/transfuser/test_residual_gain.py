"""Tests for the residual gain.

The properties that matter are the ones the ablation ladder depends on: a
freshly built gain must leave the block it sits in numerically unchanged, and it
must stay inside its stated range once it is not fresh.
"""

import torch

from lead.policy.transfuser.encoder.residual_gain import ResidualGain


def test_starts_as_an_exact_no_op() -> None:
    """A zero-initialised gain returns exactly one for every token."""
    gain = ResidualGain(n_embd=64)
    x = torch.randn(3, 17, 64)
    assert torch.allclose(gain(x), torch.ones(3, 17, 1))


def test_a_gained_block_starts_where_an_ungained_one_does() -> None:
    """Multiplying by a fresh gain leaves the attention output untouched."""
    gain = ResidualGain(n_embd=32)
    x = torch.randn(2, 9, 32)
    attention_output = torch.randn(2, 9, 32)
    assert torch.allclose(attention_output * gain(x), attention_output)


def test_stays_inside_its_range() -> None:
    """However the projection is driven, the gain is bounded by [0, max]."""
    gain = ResidualGain(n_embd=8, max_gain=2.0)
    torch.nn.init.normal_(gain.head.weight, std=50.0)
    torch.nn.init.normal_(gain.head.bias, std=50.0)
    values = gain(torch.randn(4, 11, 8) * 50.0)
    assert bool((values >= 0.0).all())
    assert bool((values <= 2.0).all())


def test_reset_restores_the_no_op() -> None:
    """Reset undoes a generic initialisation that overwrote the zeros."""
    gain = ResidualGain(n_embd=16)
    torch.nn.init.normal_(gain.head.weight, std=1.0)
    torch.nn.init.normal_(gain.head.bias, std=1.0)
    gain.reset_parameters()
    assert torch.allclose(gain(torch.randn(2, 5, 16)), torch.ones(2, 5, 1))


def test_one_scalar_per_token() -> None:
    """The gain is per token, not per channel: modality composition is the
    gate's job and magnitude is this module's."""
    gain = ResidualGain(n_embd=24)
    assert gain(torch.randn(5, 13, 24)).shape == (5, 13, 1)
