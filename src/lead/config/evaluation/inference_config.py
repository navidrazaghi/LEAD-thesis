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
    #
    # Set together with caution_risk_speed_mps below, because the pair decides
    # whether the calibrator can control the risk it is adapting against at
    # all. At 0.40 with a 2 m/s risk speed, a policy driving faster than about
    # 5 m/s cannot get under the threshold even at full caution, so risk fires
    # on every tick, the scalar pins at its ceiling and the calibration is a
    # slow switch rather than a calibration. Simulated over the loop, 0.30 with
    # a 5 m/s risk speed holds the realised rate on target from 6 to 14 m/s and
    # correctly stays off below that.
    caution_speed_floor: float = 0.3
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
    #
    # The speed here has to be reachable by slowing, or the calibrator is
    # adapting against something it cannot move: at 2 m/s nothing above about
    # 5 m/s nominal can get under it even at the floor, and the scalar simply
    # saturates. Five is the value that leaves the loop closed across the speed
    # range the benchmark actually drives at; see caution_speed_floor.
    caution_risk_threshold: float = 0.5
    caution_risk_speed_mps: float = 5.0
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
    # File the governor appends one line to per route, recording what the
    # calibrator converged to. None logs it instead of writing it.
    #
    # A field of its own rather than a reuse of evaluation.save_path, for two
    # reasons that both bite. That path is a derived property fed by an
    # environment variable, so passing it in a config dotlist raises before the
    # agent is ever built -- which reads as "Agent couldn't be set up" and
    # cost a twenty-route calibration run that failed identically every time.
    # And it points at the harness's per-route scratch directory, which is
    # wiped before each route, so anything written there would not survive to
    # be read.
    caution_calibration_log: str | None = None
    # How far the camera's predicted depth and a BEV cell's true range may
    # differ before the two modalities count as contradicting each other. Wide
    # enough to absorb the depth head's own error and the fact that a cell is
    # projected at one reference height rather than as a volume; narrow enough
    # that a sensor which has stopped seeing a surface still shows up.
    caution_depth_tolerance_meter: float = 2.5
    # Which caution signal drives the governor. "observability" reads the
    # trained head and is blind to single-modality failure by construction;
    # "cross_modal" compares the depth head against the LiDAR returns, needs no
    # label at all, and is informative exactly where the other is not.
    caution_signal: str = "observability"
    # Ensemble spread under intact sensors, in meters. Members never agree
    # exactly, so a trained ensemble has an irreducible floor -- measured at
    # 0.124 m for the rung-4 ensemble -- and caution has to respond to the
    # excess over that floor rather than to the absolute spread. Reading the
    # absolute value instead put the whole trained range, 0.12 to 0.20 m, into
    # a caution of 0.06 to 0.10, so even both sensors destroyed bought a six
    # percent reduction in speed: a signal that moved correctly and did
    # nothing.
    #
    # Both this and the scale below are properties of a particular trained
    # ensemble, not of the architecture. Re-measure them for a new checkpoint
    # with scripts/common/ensemble_spread_range.py, which prints the pair it
    # recommends, and take the baseline from the calibration route set so it is
    # not fitted to the routes the governor is then scored on.
    # Ticks the caution signal is averaged over before it is acted on. The
    # ensemble needs this to be usable at all: measured over 240 frames, its
    # spread under intact sensors has a mean of 0.124 m and a ninetieth
    # percentile of 0.254 m, while destroying both sensors moves the mean only
    # to 0.199 m. Frame-to-frame variation on a clean scene is therefore larger
    # than the shift the fault causes, and no per-tick threshold can separate
    # them. What the fault moves is the mean, and averaging shrinks the
    # variation around it as one over the square root of the window, so ten
    # ticks -- half a second at the simulator's rate -- is about what it takes.
    # One disables smoothing.
    caution_smoothing_ticks: int = 10
    caution_spread_baseline_meter: float = 0.124
    # How much excess spread over that baseline counts as fully unable to agree.
    # Measured excess at full joint degradation is 0.074 m, so this puts that
    # condition near the top of the range and leaves everything milder near the
    # bottom.
    caution_spread_meter: float = 0.075
    # Members the waypoint ensemble carries, and decoder layers each one gets.
    # Four shallow readouts rather than one deep one: the quantity wanted is
    # the disagreement between independent answers, and depth buys accuracy
    # that the frozen features already determine.
    caution_ensemble_members: int = 4
    caution_ensemble_layers: int = 1
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
