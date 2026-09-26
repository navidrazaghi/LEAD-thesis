"""A small ensemble of waypoint readouts, for measuring when they disagree.

The caution governor needs to know when the model is out of its depth. The
observability head answers that per modality and is blind to a single failed
sensor by construction; the cross-modal check answers it by comparing the two
sensors and is blind when both are wrong in the same way. Neither says anything
about a scene that is perfectly visible and simply unlike anything in training.

Disagreement between several readouts of the same features does say something
about that. Where the frozen features determine the plan, members trained on the
same data land in the same place; where they leave it ambiguous, each member
resolves the ambiguity its own way and the spread opens up. The spread is
therefore a measure of what the features fail to pin down, which is what
"epistemic" means here.

Everything upstream is frozen, so this costs one forward through a small head
rather than a retrain. Each member owns a query, a single decoder layer and a
linear readout, all reading the same context tokens the planning decoder reads.
That is the planning decoder's own structure with its depth reduced, chosen so
that members can genuinely diverge: a bare linear readout over identical inputs
and an identical loss has one least-squares answer, every member finds it, and
the spread would be a measure of nothing.

Two things supply the divergence and both are load-bearing. Members are
initialised independently, and each sees a bootstrap resample of every batch, so
they are fitted to different data rather than to the same data from different
starts. Without the resample the ensemble still trains, still reports a number,
and that number quietly shrinks towards zero as they converge on each other.

Read this in evaluation mode. The decoder layer carries dropout, so in training
mode even members with identical weights return different plans -- a different
mask per forward, not a different opinion. Sampling one network's dropout is a
real uncertainty method and a different one; a governor left in training mode
would be reading it while believing it was reading an ensemble.
"""

import jaxtyping as jt
import torch
from torch import nn

from lead.config import LeadConfig

# Feed-forward width of a member's decoder layer, as a multiple of the model
# width. Small on purpose; see the note where the layer is built.
_FEEDFORWARD_RATIO = 2


class EnsembleMember(nn.Module):
    """One readout: its own query, decoder layer and linear head."""

    def __init__(self, lead_config: LeadConfig, num_layers: int) -> None:
        """Build a member over the planning decoder's context tokens.

        Args:
            lead_config: Root config tree.
            num_layers: Decoder layers this member gets.
        """
        super().__init__()
        config = lead_config.policy.transfuser
        dimension = config.transfuser_token_dim
        self.num_waypoints = config.num_ego_pose_prediction

        self.query = nn.Parameter(
            torch.zeros(1, self.num_waypoints, dimension),
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                dimension,
                config.transfuser_num_bev_cross_attention_heads,
                # PyTorch defaults this to 2048 regardless of the model width,
                # which is where almost all of a decoder layer's parameters
                # live. Four members at that default cost more than eight
                # percent of the whole policy, which is not the "second opinion
                # off frozen features" this is meant to be. The features these
                # read are already the planning context; what a member needs is
                # enough capacity to resolve an ambiguity differently, not
                # enough to learn the task again.
                dim_feedforward=_FEEDFORWARD_RATIO * dimension,
                activation=nn.GELU(),
                batch_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(dimension),
        )
        self.head = nn.Linear(dimension, 2)
        nn.init.uniform_(self.query)

    def forward(
        self,
        context_tokens: jt.Float[torch.Tensor, "bs tokens dim"],
    ) -> jt.Float[torch.Tensor, "bs waypoints 2"]:
        """Read a plan out of the frozen context.

        Args:
            context_tokens: The planning context the frozen model produced.

        Returns:
            The member's predicted waypoints, cumulative as the planning
            decoder's are, so the two are in the same units and frame.
        """
        batch_size = context_tokens.shape[0]
        queries = self.decoder(self.query.repeat(batch_size, 1, 1), context_tokens)
        return torch.cumsum(self.head(queries), 1)


class WaypointEnsemble(nn.Module):
    """Several independent readouts of one frozen planning context."""

    def __init__(
        self,
        lead_config: LeadConfig,
        num_members: int = 4,
        num_layers: int = 1,
    ) -> None:
        """Build the ensemble.

        Args:
            lead_config: Root config tree.
            num_members: How many readouts to carry.
            num_layers: Decoder layers each member gets.

        Raises:
            ValueError: If fewer than two members are asked for, which would
                make a spread undefined rather than small.
        """
        super().__init__()
        if num_members < 2:
            raise ValueError(
                f"An ensemble needs at least two members to have a spread, got "
                f"{num_members}.",
            )
        self.members = nn.ModuleList(
            EnsembleMember(lead_config, num_layers) for _ in range(num_members)
        )

    def forward(
        self,
        context_tokens: jt.Float[torch.Tensor, "bs tokens dim"],
    ) -> jt.Float[torch.Tensor, "bs members waypoints 2"]:
        """Read every member's plan out of the same context.

        Args:
            context_tokens: The planning context the frozen model produced.

        Returns:
            Every member's waypoints, stacked on a member axis.
        """
        return torch.stack(
            [member(context_tokens) for member in self.members],
            dim=1,
        )


def bootstrap_weights(
    num_members: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> jt.Float[torch.Tensor, "members bs"]:
    """Per-member sample weights, so members fit different data.

    Poisson(1) counts are the online form of a bootstrap resample: in
    expectation a member sees each sample once, and the variation between
    members is the variation between resamples of one dataset.

    A member can draw all zeros on a small batch. That is left with a uniform
    fallback rather than allowed through, because a member with no weight
    contributes nothing to the loss for that step, and a member that keeps
    drawing nothing is a member that never trains.

    Args:
        num_members: Members to weight.
        batch_size: Samples in the batch.
        device: Device to build the weights on.
        generator: Draws the counts, so a run can repeat itself.

    Returns:
        One non-negative weight per member per sample, with no member left
        entirely unweighted.
    """
    rates = torch.ones(num_members, batch_size, device=device)
    weights = torch.poisson(rates, generator=generator)
    empty = weights.sum(dim=1) == 0.0
    if bool(empty.any()):
        weights[empty] = 1.0
    return weights


def ensemble_loss(
    predictions: jt.Float[torch.Tensor, "bs members waypoints 2"],
    label: jt.Float[torch.Tensor, "bs waypoints 2"],
    weights: jt.Float[torch.Tensor, "members bs"] | None = None,
) -> jt.Float[torch.Tensor, ""]:
    """Weighted L1 of every member against the expert's plan.

    Every member appears in this sum, which is what keeps the first-step
    gradient check satisfied: a member left out of the loss would carry
    trainable parameters that never receive a gradient, and training refuses to
    start in that state.

    Args:
        predictions: Every member's waypoints.
        label: The expert's waypoints, shared by all members.
        weights: Per-member, per-sample bootstrap weights; uniform when None.

    Returns:
        The mean weighted error.
    """
    error = (predictions - label.unsqueeze(1)).abs().mean(dim=(2, 3))
    if weights is None:
        return error.mean()
    per_member = weights.transpose(0, 1)
    return (error * per_member).sum() / per_member.sum().clamp(min=1.0)


def ensemble_spread(
    predictions: jt.Float[torch.Tensor, "bs members waypoints 2"],
) -> jt.Float[torch.Tensor, " bs"]:
    """How far apart the members' plans are, per sample, in meters.

    The spread is taken over members at each waypoint and then averaged over
    waypoints, rather than the other way round: members that agree early and
    diverge late describe a plan that is uncertain, and averaging the plans
    first would hide exactly that.

    Args:
        predictions: Every member's waypoints.

    Returns:
        One spread per sample.
    """
    # Population standard deviation over members: with four of them the
    # sample correction is a 15% inflation that means nothing here, and the
    # quantity wanted is the spread of the members present.
    per_waypoint = predictions.float().std(dim=1, unbiased=False)
    return per_waypoint.norm(dim=-1).mean(dim=-1)
