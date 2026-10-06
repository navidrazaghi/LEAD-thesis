"""Tests for the weather-visibility head.

The head exists to close an asymmetry that is already in the repository: the
expert reads the visibility class and drives differently under it, and the
student imitates those decisions without seeing the variable. What these check
is that the head is reachable, that it is silent while switched off, and that
it accepts the label in whatever shape the collate hands over -- the key
travels through the batch verbatim, so its type is not this head's to assume.
"""

import pytest
import torch

from lead.config import LeadConfig
from lead.policy.transfuser.decoder.visibility_decoder import (
    NUM_VISIBILITY_CLASSES,
    VisibilityDecoder,
)


def _config(enabled: bool) -> LeadConfig:
    """A config with the head on or off."""
    lead_config = LeadConfig()
    lead_config.policy.transfuser.use_weather_visibility = enabled
    return lead_config


def _grid(lead_config: LeadConfig, batch: int = 3) -> torch.Tensor:
    """A BEV feature grid of the width the encoder produces."""
    channels = lead_config.policy.transfuser.bev_feature_channels
    return torch.randn(batch, channels, 8, 8)


class TestShape:
    """One score per class per frame, whatever the grid size."""

    def test_one_logit_per_class(self) -> None:
        """Four classes: CLEAR, OK, LIMITED, VERY_LIMITED."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        assert head(_grid(lead_config)).shape == (3, NUM_VISIBILITY_CLASSES)

    def test_grid_size_does_not_reach_the_output(self) -> None:
        """Visibility is a property of the frame, so the grid is pooled away."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        channels = lead_config.policy.transfuser.bev_feature_channels
        wide = head(torch.randn(2, channels, 20, 30))
        narrow = head(torch.randn(2, channels, 5, 4))
        assert wide.shape == narrow.shape == (2, NUM_VISIBILITY_CLASSES)


class TestLoss:
    """Silent while off; a cross entropy while on."""

    def test_off_writes_no_loss(self) -> None:
        """A run that did not ask for it must not pay for it."""
        lead_config = _config(False)
        head = VisibilityDecoder(lead_config)
        loss: dict = {}
        head.compute_loss(
            torch.randn(2, NUM_VISIBILITY_CLASSES),
            {"visual_visibility": torch.tensor([0, 3])},
            loss,
            {},
        )
        assert loss == {}

    def test_on_writes_a_finite_loss(self) -> None:
        """The head is trainable once asked for."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        loss: dict = {}
        head.compute_loss(
            torch.randn(4, NUM_VISIBILITY_CLASSES),
            {"visual_visibility": torch.tensor([0, 1, 2, 3])},
            loss,
            {},
        )
        assert torch.isfinite(loss["loss_weather_visibility"])

    def test_a_confident_correct_prediction_costs_almost_nothing(self) -> None:
        """The loss points the way it should."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        right, wrong = {}, {}
        scores = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
        head.compute_loss(scores, {"visual_visibility": torch.tensor([0])}, right, {})
        head.compute_loss(scores, {"visual_visibility": torch.tensor([3])}, wrong, {})
        assert right["loss_weather_visibility"] < 0.01
        assert wrong["loss_weather_visibility"] > 5.0

    def test_accuracy_is_logged(self) -> None:
        """A diagnostic worth watching per epoch, as the gate's values are."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        log: dict = {}
        head.compute_loss(
            torch.tensor([[9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 9.0]]),
            {"visual_visibility": torch.tensor([0, 3])},
            {},
            log,
        )
        assert log["weather_visibility_accuracy"] == pytest.approx(1.0)


class TestLabelShapes:
    """The key travels verbatim, so its type is not ours to assume."""

    @pytest.mark.parametrize("label", [
        [0, 2],
        (0, 2),
        torch.tensor([0, 2]),
        torch.tensor([[0], [2]]),
    ])
    def test_the_label_is_accepted_however_it_arrives(self, label: object) -> None:
        """A list, a tuple, a flat tensor, or a column of them."""
        lead_config = _config(True)
        head = VisibilityDecoder(lead_config)
        loss: dict = {}
        head.compute_loss(
            torch.randn(2, NUM_VISIBILITY_CLASSES),
            {"visual_visibility": label},
            loss,
            {},
        )
        assert torch.isfinite(loss["loss_weather_visibility"])
