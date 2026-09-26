"""Tests for the waypoint ensemble the caution governor reads disagreement from.

The failure this guards against is not a crash. An ensemble whose members
converge on each other trains fine, reports a number every tick, and that number
is quietly zero -- so the governor stays inert and the run looks like a clean
negative result rather than a broken signal. The tests that matter are therefore
the ones about whether the members can disagree at all, and about the two
mechanisms that keep them apart.
"""

import pytest
import torch

from lead.config import LeadConfig
from lead.policy.transfuser.decoder.waypoint_ensemble import (
    EnsembleMember,
    WaypointEnsemble,
    bootstrap_weights,
    ensemble_loss,
    ensemble_spread,
)


@pytest.fixture(name="config")
def _config() -> LeadConfig:
    return LeadConfig()


def _context(config: LeadConfig, batch_size: int = 3, tokens: int = 12) -> torch.Tensor:
    dimension = config.policy.transfuser.transfuser_token_dim
    return torch.randn(batch_size, tokens, dimension)


class TestShapes:
    """What the ensemble produces, so the caution reading can index it."""

    def test_a_member_predicts_the_configured_waypoints(
        self,
        config: LeadConfig,
    ) -> None:
        member = EnsembleMember(config, num_layers=1)
        output = member(_context(config))
        assert output.shape == (3, config.policy.transfuser.num_ego_pose_prediction, 2)

    def test_the_ensemble_stacks_members_on_their_own_axis(
        self,
        config: LeadConfig,
    ) -> None:
        ensemble = WaypointEnsemble(config, num_members=4, num_layers=1)
        output = ensemble(_context(config))
        assert output.shape == (
            3,
            4,
            config.policy.transfuser.num_ego_pose_prediction,
            2,
        )

    def test_one_member_is_refused(self, config: LeadConfig) -> None:
        with pytest.raises(ValueError, match="at least two members"):
            WaypointEnsemble(config, num_members=1)

    def test_a_member_stays_within_its_parameter_budget(
        self,
        config: LeadConfig,
    ) -> None:
        """This is meant to be a second opinion, not a second model.

        PyTorch defaults a decoder layer's feed-forward width to 2048 whatever
        the model width is, which doubled the cost of the whole ensemble before
        it was pinned down. The bound is loose enough not to break on a
        reasonable change and tight enough to catch that default coming back.
        """
        member = EnsembleMember(config, num_layers=1)
        count = sum(parameter.numel() for parameter in member.parameters())
        assert count < 1_000_000, f"{count:,} parameters in one member"


class TestMembersCanActuallyDisagree:
    """The property the whole signal rests on."""

    def test_independently_initialised_members_do_not_agree(
        self,
        config: LeadConfig,
    ) -> None:
        torch.manual_seed(0)
        ensemble = WaypointEnsemble(config, num_members=4, num_layers=1).eval()
        spread = ensemble_spread(ensemble(_context(config)))
        assert bool((spread > 1e-3).all())

    def test_identical_members_report_no_spread(self, config: LeadConfig) -> None:
        """The degenerate case, stated so its signature is recognisable.

        If the members ever end up identical -- shared initialisation, or
        convergence onto one another -- this is what the governor would see:
        exactly zero, on every frame, indistinguishable from confidence.
        """
        torch.manual_seed(0)
        ensemble = WaypointEnsemble(config, num_members=3, num_layers=1).eval()
        reference = ensemble.members[0].state_dict()
        for member in ensemble.members[1:]:
            member.load_state_dict(reference)
        spread = ensemble_spread(ensemble(_context(config)))
        assert bool((spread < 1e-5).all())

    def test_dropout_must_not_be_read_as_disagreement(
        self,
        config: LeadConfig,
    ) -> None:
        """Why the governor has to read this ensemble in eval mode.

        The decoder layer carries dropout, so in training mode even identical
        members produce different plans -- a different mask per forward, not a
        different opinion. Sampling one network's dropout is a real uncertainty
        method, but it is a different one, and a governor left in training mode
        would be reading it while believing it was reading an ensemble.
        """
        torch.manual_seed(0)
        ensemble = WaypointEnsemble(config, num_members=3, num_layers=1)
        reference = ensemble.members[0].state_dict()
        for member in ensemble.members[1:]:
            member.load_state_dict(reference)
        context = _context(config)

        with torch.no_grad():
            ensemble.train()
            noise = float(ensemble_spread(ensemble(context)).mean())
            ensemble.eval()
            opinion = float(ensemble_spread(ensemble(context)).mean())

        assert noise > 0.1, "dropout should move identical members in train mode"
        assert opinion < 1e-5, "identical members must agree once dropout is off"

    def test_spread_grows_as_members_are_pulled_apart(
        self,
        config: LeadConfig,
    ) -> None:
        base = torch.zeros(2, 4, 8, 2)
        previous = -1.0
        for offset in (0.0, 0.5, 1.0, 2.0):
            predictions = base.clone()
            predictions[:, 0] += offset
            spread = float(ensemble_spread(predictions).mean())
            assert spread > previous
            previous = spread


class TestBootstrap:
    """The second divergence mechanism, and the trap inside it."""

    def test_members_get_different_weights(self) -> None:
        torch.manual_seed(0)
        weights = bootstrap_weights(4, 32, torch.device("cpu"))
        assert weights.shape == (4, 32)
        assert not torch.equal(weights[0], weights[1])

    def test_weights_are_non_negative(self) -> None:
        torch.manual_seed(0)
        weights = bootstrap_weights(4, 32, torch.device("cpu"))
        assert bool((weights >= 0.0).all())

    def test_no_member_is_left_entirely_unweighted(self) -> None:
        """A member with no weight contributes nothing and never trains.

        On a small batch a Poisson draw can come back all zeros, which is rare
        enough to survive a smoke test and frequent enough to matter over an
        epoch.
        """
        for seed in range(50):
            torch.manual_seed(seed)
            weights = bootstrap_weights(6, 2, torch.device("cpu"))
            assert bool((weights.sum(dim=1) > 0.0).all()), seed

    def test_a_seeded_draw_repeats(self) -> None:
        first = bootstrap_weights(
            4, 16, torch.device("cpu"), torch.Generator().manual_seed(7),
        )
        second = bootstrap_weights(
            4, 16, torch.device("cpu"), torch.Generator().manual_seed(7),
        )
        assert torch.equal(first, second)


class TestLoss:
    """Training every member, which is what the gradient check demands."""

    def test_every_member_receives_a_gradient(self, config: LeadConfig) -> None:
        """Training refuses to start if a trainable parameter takes none."""
        torch.manual_seed(0)
        ensemble = WaypointEnsemble(config, num_members=4, num_layers=1)
        waypoints = config.policy.transfuser.num_ego_pose_prediction
        predictions = ensemble(_context(config))
        label = torch.randn(3, waypoints, 2)
        ensemble_loss(predictions, label).backward()
        starved = [
            name
            for name, parameter in ensemble.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not starved, starved

    def test_a_perfect_prediction_costs_nothing(self, config: LeadConfig) -> None:
        label = torch.randn(2, 8, 2)
        predictions = label.unsqueeze(1).repeat(1, 4, 1, 1)
        assert float(ensemble_loss(predictions, label)) == pytest.approx(0.0)

    def test_weights_change_which_samples_count(self) -> None:
        label = torch.zeros(2, 4, 2)
        predictions = torch.zeros(2, 2, 4, 2)
        predictions[0] = 10.0
        ignore_first = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        assert float(ensemble_loss(predictions, label, ignore_first)) == pytest.approx(
            0.0,
        )

    def test_uniform_weights_match_the_unweighted_loss(self) -> None:
        torch.manual_seed(0)
        label = torch.randn(3, 4, 2)
        predictions = torch.randn(3, 2, 4, 2)
        uniform = torch.ones(2, 3)
        assert float(ensemble_loss(predictions, label, uniform)) == pytest.approx(
            float(ensemble_loss(predictions, label)),
            rel=1e-5,
        )


class TestSpreadMetric:
    """Properties of the number the governor reads."""

    def test_agreement_is_exactly_zero(self) -> None:
        predictions = torch.randn(2, 1, 5, 2).repeat(1, 4, 1, 1)
        assert bool((ensemble_spread(predictions) < 1e-6).all())

    def test_it_is_reported_per_sample(self) -> None:
        assert ensemble_spread(torch.randn(7, 4, 5, 2)).shape == (7,)

    def test_it_is_never_negative(self) -> None:
        torch.manual_seed(0)
        assert bool((ensemble_spread(torch.randn(5, 4, 6, 2)) >= 0.0).all())

    def test_late_divergence_is_not_averaged_away(self) -> None:
        """Members that agree at first and part later describe an uncertain plan."""
        agree_throughout = torch.zeros(1, 2, 6, 2)
        part_later = torch.zeros(1, 2, 6, 2)
        part_later[0, 1, 3:] = 3.0
        assert float(ensemble_spread(part_later)) > float(
            ensemble_spread(agree_throughout),
        )
