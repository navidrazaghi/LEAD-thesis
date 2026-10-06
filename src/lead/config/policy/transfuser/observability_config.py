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
    # Which quantity the gate is pulled towards. "logit" is what every result
    # in this project was trained under: sigmoid on the gate output, binary
    # cross entropy against observability, which converges on the log odds.
    # "log" is what the inverse-variance derivation actually prescribes, the
    # log of observability, and differs from the log odds by -log(1 - v) --
    # negligible for a blind modality, 2.3 nats for one at 0.9. Only the
    # difference between the two modalities of a token can matter, since the
    # softmax is shift invariant, so the "log" objective compares centred
    # values and a token with one supervised modality contributes nothing.
    observability_gate_target: str = "logit"
    # Predict the weather visibility class the expert conditions on. The label
    # is in every batch already and no run has read it; the expert changes its
    # lane-change transition and its target speed under LIMITED and
    # VERY_LIMITED, so the student currently imitates those decisions without
    # seeing what caused them. Off by default: no result here was trained with
    # it.
    use_weather_visibility: bool = False
    # Weight of that head relative to the other tasks. The per-task weights are
    # normalized by their sum, so this quietly shrinks the driving losses like
    # every other auxiliary task does.
    weather_visibility_loss_weight: float = 1.0
    # Width of the head's single hidden layer.
    weather_visibility_head_channels: int = 64
    # Scales how much of the attention output enters the residual stream, per
    # token. The gate decides which modality a query reads from; this decides
    # how much of that read survives into the token, which the gate cannot
    # touch. Zero-initialized, so turning it on does not move the starting
    # point. Needs a deformable backbone for the same reason the gate does.
    use_residual_gain: bool = False
    # Carry several independent readouts of the planning context, so their
    # disagreement can be read as a caution signal. Unlike the observability
    # head this needs no label, and unlike the cross-modal check it says
    # something about a scene that is perfectly visible and simply unlike
    # anything in training.
    use_waypoint_ensemble: bool = False
    # Weight of the ensemble's supervision relative to the other tasks. The
    # intended use is a fine-tune with everything else frozen, where nothing is
    # left to dilute; at full weight in a joint run this would shrink the
    # driving losses like any other task, because the weights are normalized by
    # their sum.
    waypoint_ensemble_loss_weight: float = 1.0
