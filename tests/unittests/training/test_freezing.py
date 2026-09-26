"""Tests for freezing everything but a named set of parameters.

The failure this guards against is the quiet one. A prefix that matches nothing
freezes the whole model, and a run in that state looks entirely healthy: the
loss is computed, the steps tick by, checkpoints are written, and the only thing
that would have told you is a number in one log line. So the mismatch is an
error rather than a warning, and the message says what the names actually are.
"""

import pytest
import torch
from torch import nn

from lead.training.train import _freeze_all_but


def _model() -> nn.Module:
    """A stand-in with the shape that matters: several named submodules."""
    return nn.ModuleDict(
        {
            "backbone": nn.Linear(4, 4),
            "planning_decoder": nn.Linear(4, 4),
            "waypoint_ensemble": nn.Linear(4, 2),
        },
    )


def _trainable(model: nn.Module) -> set[str]:
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


class TestFreezing:
    """What is left trainable."""

    def test_an_empty_selection_trains_everything(self) -> None:
        """The default, which is what every existing rung relies on."""
        model = _model()
        _freeze_all_but(model, ())
        assert _trainable(model) == {
            name for name, _ in model.named_parameters()
        }

    def test_only_the_named_module_stays_trainable(self) -> None:
        model = _model()
        _freeze_all_but(model, ("waypoint_ensemble",))
        assert _trainable(model) == {
            "waypoint_ensemble.weight",
            "waypoint_ensemble.bias",
        }

    def test_several_prefixes_are_all_kept(self) -> None:
        model = _model()
        _freeze_all_but(model, ("waypoint_ensemble", "planning_decoder"))
        assert _trainable(model) == {
            "waypoint_ensemble.weight",
            "waypoint_ensemble.bias",
            "planning_decoder.weight",
            "planning_decoder.bias",
        }

    def test_a_frozen_parameter_takes_no_gradient(self) -> None:
        """Which is also what keeps the first-step gradient check satisfied.

        That check raises on a trainable parameter with no gradient, and reads
        requires_grad to decide what counts, so freezing is the supported way
        to leave a module out of the optimisation.
        """
        model = _model()
        _freeze_all_but(model, ("waypoint_ensemble",))
        output = model["waypoint_ensemble"](model["backbone"](torch.randn(2, 4)))
        output.sum().backward()
        assert model["backbone"].weight.grad is None
        assert model["waypoint_ensemble"].weight.grad is not None


class TestTypos:
    """A prefix that matches nothing is the dangerous case."""

    def test_a_name_matching_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="match no parameter"):
            _freeze_all_but(_model(), ("waypoint_ensembl",))

    def test_a_truncated_name_does_not_match_by_string_prefix(self) -> None:
        """The typo that a raw prefix test would let through.

        "waypoint_ensembl" is a string prefix of "waypoint_ensemble.weight", so
        matching on characters alone would accept the typo, keep the right
        module trainable, and leave the mismatch check with nothing to report.
        Matching at the dot is what makes the check above able to fire at all.
        """
        with pytest.raises(ValueError, match="waypoint_ensembl"):
            _freeze_all_but(_model(), ("waypoint_ensembl",))

    def test_an_exact_parameter_name_is_accepted(self) -> None:
        """Freezing down to a single parameter, not just a whole module."""
        model = _model()
        _freeze_all_but(model, ("waypoint_ensemble.weight",))
        assert _trainable(model) == {"waypoint_ensemble.weight"}

    def test_the_message_lists_the_names_that_do_exist(self) -> None:
        with pytest.raises(ValueError, match="backbone"):
            _freeze_all_but(_model(), ("nonsense",))

    def test_one_bad_prefix_among_good_ones_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="nonsense"):
            _freeze_all_but(_model(), ("waypoint_ensemble", "nonsense"))
