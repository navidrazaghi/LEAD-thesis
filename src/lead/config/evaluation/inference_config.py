"""Inference configuration: checkpoint loading and prediction post-processing."""

from lead.config.node import ConfigNode


class InferenceConfig(ConfigNode):
    """Prediction post-processing knobs and which output modality drives each control."""

    # --- Model Settings ---
    # If true lower the target speed using a factor
    lower_target_speed: bool = False
    # Factor to multiply the target speed with when lowering is enabled
    lower_target_speed_factor: float = 0.8
    # Confidence threshold for brake action (full brake applied if confidence exceeds this)
    brake_threshold: float = 0.9

    # --- Caution governor ---
    # If true, scale the predicted target speed by how well the frozen model
    # says it resolves the road ahead. The network is untouched: this reads the
    # observability head the model already computes and acts on it in the
    # controller, which is the one place its authority is not fractional.
    use_caution_governor: bool = False
    # Slowest the governor may drive, as a fraction of the predicted speed. A
    # floor rather than a free hand: the calibrator bounds a long-run average
    # and says nothing about one tick, so it must not be able to stop the car.
    caution_speed_floor: float = 0.4
    # How the two modalities combine into one per-cell confidence.
    #
    # "best" is the default because it is what the evidence supports: the
    # trained rungs score no worse under full camera destruction than intact,
    # so one working sensor is enough and slowing down for a single failed one
    # would cost route completion to buy nothing. The consequence has to be
    # stated, because it decides what this can be evaluated on -- measured on
    # the recorded per-modality means, "best" moves caution by only about 0.02
    # between intact and full single-modality destruction, so under every
    # condition this project has run so far the governor is deliberately
    # inert. Its regime is joint degradation, the "both" condition, where
    # redundancy has nothing left to fall back on.
    #
    # "mean" and "worst" trade that away for a signal that moves under
    # single-modality damage (about 0.41 and 0.81 respectively). They are here
    # so the choice is measured rather than argued.
    caution_modality_rule: str = "best"
    # The corridor the caution is measured over: the road the ego is about to
    # drive through, rather than the whole grid, where cells behind the car
    # would dominate the mean and being unable to see costs nothing.
    caution_corridor_length_meter: float = 20.0
    caution_corridor_half_width_meter: float = 4.0
    # Surrogate risk: carrying speed into a stretch the model cannot resolve.
    # Infractions are what matter but are too rare and too late to calibrate on.
    caution_risk_threshold: float = 0.5
    caution_risk_speed_mps: float = 2.0
    # Long-run rate of surrogate risk events the calibrator converges to, and
    # how fast it moves. The target is the knob a reader can reason about; the
    # threshold it implies is what gets adapted.
    caution_target_risk: float = 0.05
    caution_step_size: float = 0.05
    caution_ceiling: float = 1.0
    # Where the scalar starts each route. Zero means the governor begins inert
    # and adapts from scratch, which leaves the opening stretch of every scored
    # route un-governed. To avoid that, run the governor over the calibration
    # route set -- routes the scored sets do not use, so nothing is fitted to
    # its own test -- and set this to the scalar it converged to. It is a
    # starting point rather than a commitment: the calibrator keeps adapting
    # during the scored route, so a warm start that was wrong is corrected.
    caution_initial_lambda: float = 0.0
    # If true be strict when load weight
    # --- Sensor degradation at inference ---
    # Which modality to damage while driving: "camera", "lidar" or "none".
    # This is how the robustness claim is measured: a sweep of severities at a
    # fixed modality traces the degradation curve, where the training-time
    # curriculum instead randomises both per sample.
    degrade_modality: str = "none"
    # How badly to damage it, in [0, 1]. Zero is a no-op whatever the modality.
    degrade_severity: float = 0.0
    # Seeds the damage. Two checkpoints driven over one route with the same
    # seed meet the same noise and the same dropped returns, so the difference
    # between them is the model rather than two draws from one distribution.
    degrade_seed: int = 0

    strict_weight_load: bool = True

    # --- Image Processing ---
    # JPEG quality used in inference (0-100)
    jpeg_quality: int = 90

    # --- Control which output is used for controlling ---
    # Modality used for steering control
    steer_modality: str = "route"
    # Modality used for throttle control
    throttle_modality: str = "target_speed"
    # Modality used for brake control
    brake_modality: str = "target_speed"
