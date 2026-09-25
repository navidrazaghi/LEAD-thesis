"""Per-cell, per-modality observability supervision of the TransFuser model."""

from lead.config.node import ConfigNode


class TransfuserObservabilityConfig(ConfigNode):
    """The auxiliary head predicting how much each modality can see, where.

    The expert decides which actors to react to by counting each one's LiDAR
    returns and visible camera pixels against a per-class threshold (see
    :class:`~lead.config.expert.occlusion_config.ExpertOcclusionConfig`). That
    check runs on the privileged side only; the student never learns it. This
    head is its twin on the student side, so the policy carries an explicit
    estimate of what its own sensors can resolve rather than treating every
    region of every modality as equally informative.
    """

    # If true run the observability head and its loss. Like the other head
    # toggles it also decides whether the targets reach the cache store, so
    # turning it on needs a store built with it on.
    use_observability: bool = False
    # If true the target is the count over its threshold, clipped to one, so a
    # barely-seen actor supervises a lower value than a fully visible one. If
    # false the target is the expert's own binary decision.
    observability_soft_targets: bool = True
    # Feature width of the observability head.
    observability_head_channels: int = 64

    # If true the deformable fusion's modality weights are shifted by a
    # per-token observability gate, so a query reads less from a modality that
    # cannot resolve its part of the scene. Needs a deformable backbone; the
    # base fusion has no modality axis to shift. The gate is zero-initialized,
    # so turning it on does not move the starting point.
    use_observability_gate: bool = False
    # Weight of the dense head's supervision relative to the other tasks.
    # Worth turning down: the per-task weights are normalized by their sum, so
    # every auxiliary task at full weight quietly shrinks the driving losses.
    observability_loss_weight: float = 1.0
    # Weight of the gate's own supervision relative to the other tasks. The
    # gate also receives gradient through the driving losses, so this only sets
    # how hard it is pulled towards the expert's measured visibility.
    observability_gate_loss_weight: float = 1.0
    # Scales how much of the attention output enters the residual stream, per
    # token. The gate decides which modality a query reads from; this decides
    # how much of that read survives into the token, which the gate cannot
    # touch. Zero-initialized, so turning it on does not move the starting
    # point. Needs a deformable backbone for the same reason the gate does.
    use_residual_gain: bool = False
