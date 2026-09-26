"""Turn the observability estimate into speed, instead of into attention weights.

The observability head was shown to be informative: its output tracks sensor
degradation, and the gate fed from the same supervision moved attention mass by
a factor of eleven to thirty-eight in the right direction. What it did not do
was drive better. The measured reason is that the gate acts on attention logits,
and attention's authority over the token that leaves a fusion block is only
about four tenths, so a large shift in what the model reads became a small shift
in what it does -- and under full camera destruction the small shift was
negative.

This actuates the same signal where the authority is not fractional. If the
model cannot resolve the road ahead, the car slows down. Nothing about the
network changes: the checkpoint is frozen, the forward pass is untouched, and
the only thing this reads is a head the model already computes.

Two choices in the signal are worth stating because they are not the obvious
ones. The modalities are combined by taking the better of the two per cell
rather than averaging them, because redundancy is the point of carrying two
sensors -- a cell one modality resolves is resolved, and averaging would call it
half-seen. And only the corridor the ego is about to drive through counts,
because a mean over the whole grid is dominated by cells behind the car and out
to the sides, where being unable to see costs nothing.
"""

import jaxtyping as jt
import torch

from lead.config import LeadConfig


def corridor_mask(
    lead_config: LeadConfig,
    cell_height: int,
    cell_width: int,
    device: torch.device,
) -> jt.Bool[torch.Tensor, "cell_h cell_w"]:
    """Which cells of the observability grid lie in the ego's driving corridor.

    Args:
        lead_config: Root config tree, for the BEV grid's extent in meters.
        cell_height: Rows of the observability grid.
        cell_width: Columns of the observability grid.
        device: Device to build the mask on.

    Returns:
        The mask, true where a cell is ahead of the ego and inside the corridor
        half-width.
    """
    config = lead_config.policy.transfuser
    governor = lead_config.evaluation.inference

    # Cell centres in meters, on the same axes the BEV raster is built on.
    row_edges = torch.linspace(
        config.bev_min_x_meter, config.bev_max_x_meter, cell_height + 1, device=device,
    )
    column_edges = torch.linspace(
        config.bev_min_y_meter, config.bev_max_y_meter, cell_width + 1, device=device,
    )
    forward = 0.5 * (row_edges[:-1] + row_edges[1:])
    lateral = 0.5 * (column_edges[:-1] + column_edges[1:])

    ahead = (forward >= 0.0) & (forward <= governor.caution_corridor_length_meter)
    within = lateral.abs() <= governor.caution_corridor_half_width_meter
    return ahead.view(-1, 1) & within.view(1, -1)


def observability_caution(
    observability_logits: jt.Float[torch.Tensor, "bs n_modalities cell_h cell_w"],
    lead_config: LeadConfig,
) -> float:
    """How poorly the sensors resolve the road ahead, in ``[0, 1]``.

    Args:
        observability_logits: The head's per-modality logits over the BEV cell
            grid, as the frozen model computed them.
        lead_config: Root config tree.

    Returns:
        Zero when the corridor is resolved, one when it is not.

    Raises:
        ValueError: If the configured modality rule is not one of the three.
    """
    resolved = torch.sigmoid(observability_logits.float())
    rule = lead_config.evaluation.inference.caution_modality_rule
    if rule == "best":
        best, _ = resolved.max(dim=1)
    elif rule == "mean":
        best = resolved.mean(dim=1)
    elif rule == "worst":
        best, _ = resolved.min(dim=1)
    else:
        raise ValueError(
            f"caution_modality_rule must be 'best', 'mean' or 'worst', got {rule!r}.",
        )

    mask = corridor_mask(
        lead_config,
        best.shape[-2],
        best.shape[-1],
        best.device,
    )
    if not bool(mask.any()):
        # A corridor that lands outside the grid would otherwise divide by zero
        # and report perfect confidence, which is the wrong way to fail.
        return 1.0
    corridor = best[..., mask]
    return float(1.0 - corridor.mean())


def target_speed_multiplier(
    caution: float,
    conservativeness: float,
    lead_config: LeadConfig,
) -> float:
    """Map caution and the calibrated conservativeness onto a speed factor.

    The calibrated scalar sets the slope rather than a cut-off: at zero the
    governor is inert and the policy drives exactly as the frozen checkpoint
    does, and as it rises the same caution buys progressively more slowing. That
    keeps the un-governed model inside the family, so the ablation compares a
    mechanism against its own absence rather than against a different tuning.

    Args:
        caution: How poorly the road ahead is resolved, in ``[0, 1]``.
        conservativeness: The calibrator's scalar; zero disables the governor.
        lead_config: Root config tree, for the floor.

    Returns:
        A multiplier in ``[floor, 1]`` to apply to the predicted target speed.
    """
    floor = lead_config.evaluation.inference.caution_speed_floor
    demand = min(max(caution * conservativeness, 0.0), 1.0)
    return float(1.0 - (1.0 - floor) * demand)


def surrogate_risk_event(
    caution: float,
    ego_speed_mps: float,
    lead_config: LeadConfig,
) -> bool:
    """Whether this tick counts as a risk event for the calibrator.

    An infraction is the thing worth avoiding but it is far too rare and far too
    late to calibrate on: a run produces a handful, and each one arrives after
    the decisions that caused it. This stands in for it with the condition those
    decisions share -- carrying speed into a stretch of road the model cannot
    resolve. It is a surrogate and is named one; what it has to be is frequent,
    observable at the tick, and monotone in the thing it stands for.

    Args:
        caution: How poorly the road ahead is resolved, in ``[0, 1]``.
        ego_speed_mps: Current speed.
        lead_config: Root config tree.

    Returns:
        True when the tick is one the calibrator should count against.
    """
    governor = lead_config.evaluation.inference
    unresolved = caution >= governor.caution_risk_threshold
    moving = ego_speed_mps >= governor.caution_risk_speed_mps
    return bool(unresolved and moving)
