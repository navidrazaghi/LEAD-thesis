"""Tests for the two objectives the modality gate can be trained against.

The first class is the one that protects the existing results: the default
objective must return exactly what it returned before the alternative existed,
because every checkpoint in this project was trained under it and an
evaluation is queued that will load this code.

The rest check the property the alternative exists for. The softmax turns a
bias difference into a mass ratio, so only the difference across the modality
axis is defined; the log objective is built to be invariant to a constant
added to both modalities, and to reproduce the ratio inverse-variance
weighting asks for rather than the larger one the log odds produces.
"""

import math

import pytest
import torch

from lead.policy.transfuser.encoder.observability_gate import gate_loss


def _one_token(camera: float, lidar: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of one token whose two modalities carry these observabilities."""
    target = torch.tensor([[[camera, lidar]]], dtype=torch.float32)
    mask = torch.ones_like(target)
    return target, mask


class TestTheDefaultIsUnchanged:
    """The objective every existing checkpoint was trained under."""

    def test_default_matches_binary_cross_entropy(self) -> None:
        """The value, not merely the shape, has to be the old one."""
        target, mask = _one_token(0.9, 0.4)
        logits = [torch.tensor([[[0.3, -0.7]]])]
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0], target, reduction="none",
        ).mean()
        assert gate_loss(logits, target, mask) == pytest.approx(
            float(expected), abs=1e-6)

    def test_the_default_is_the_logit_objective(self) -> None:
        """Naming it explicitly must not change anything."""
        target, mask = _one_token(0.7, 0.2)
        logits = [torch.tensor([[[0.1, 0.5]]])]
        assert gate_loss(logits, target, mask) == pytest.approx(
            float(gate_loss(logits, target, mask, "logit")), abs=1e-7)

    def test_an_unknown_objective_is_refused(self) -> None:
        """A typo must not silently fall back to either behaviour."""
        target, mask = _one_token(0.5, 0.5)
        with pytest.raises(ValueError, match="Unknown gate objective"):
            gate_loss([torch.zeros(1, 1, 2)], target, mask, "logodds")


class TestTheLogObjective:
    """What the inverse-variance derivation actually prescribes."""

    def test_it_is_minimised_at_the_prescribed_difference(self) -> None:
        """A gate whose difference equals the log ratio has nothing to fix."""
        target, mask = _one_token(0.9, 0.5)
        wanted = math.log(0.9) - math.log(0.5)
        exact = torch.tensor([[[wanted / 2, -wanted / 2]]])
        assert gate_loss([exact], target, mask, "log") == pytest.approx(0.0, abs=1e-6)

    def test_the_log_odds_solution_is_penalised(self) -> None:
        """The point of the change: the old optimum is not the new one."""
        target, mask = _one_token(0.9, 0.5)
        log_odds = torch.tensor([[[
            math.log(0.9 / 0.1),
            math.log(0.5 / 0.5),
        ]]])
        assert gate_loss([log_odds], target, mask, "log") > 0.1

    def test_it_ignores_a_constant_added_to_both_modalities(self) -> None:
        """The softmax ignores it, so the objective must too."""
        target, mask = _one_token(0.8, 0.3)
        logits = torch.tensor([[[0.4, -0.2]]])
        plain = gate_loss([logits], target, mask, "log")
        shifted = gate_loss([logits + 3.7], target, mask, "log")
        assert float(plain) == pytest.approx(float(shifted), abs=1e-6)

    def test_a_token_with_one_supervised_modality_contributes_nothing(self) -> None:
        """With one modality there is no difference to supervise."""
        target = torch.tensor([[[0.9, 0.4]]])
        mask = torch.tensor([[[1.0, 0.0]]])
        assert gate_loss([torch.tensor([[[5.0, -5.0]]])], target, mask,
                         "log") == pytest.approx(0.0, abs=1e-7)

    def test_a_blind_modality_does_not_produce_a_non_finite_loss(self) -> None:
        """Observability reaches zero; its log does not."""
        target, mask = _one_token(0.9, 0.0)
        value = gate_loss([torch.tensor([[[1.0, -1.0]]])], target, mask, "log")
        assert torch.isfinite(value)

    def test_nothing_supervised_gives_zero(self) -> None:
        """Same contract as the default objective."""
        target = torch.tensor([[[0.5, 0.5]]])
        mask = torch.zeros_like(target)
        assert gate_loss([torch.ones(1, 1, 2)], target, mask,
                         "log") == pytest.approx(0.0, abs=1e-7)

    def test_blocks_are_averaged(self) -> None:
        """One loss per fusion block, meaned, as before."""
        target, mask = _one_token(0.9, 0.5)
        wanted = math.log(0.9) - math.log(0.5)
        exact = torch.tensor([[[wanted / 2, -wanted / 2]]])
        wrong = torch.tensor([[[2.0, -2.0]]])
        both = gate_loss([exact, wrong], target, mask, "log")
        only = gate_loss([wrong], target, mask, "log")
        assert float(both) == pytest.approx(float(only) / 2, abs=1e-6)
