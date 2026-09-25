"""Train a policy to plan for the moment its plan is actually executed.

A deployed stack does not act on the frame it is looking at. Perception,
inference and actuation each cost time, so the waypoints applied at tick t were
computed from the observation at tick t-k. Nothing in this dataset contains that
gap: every label is the future of the frame it is attached to, so the policy
learns to plan from an observation it will never have.

The obvious way to inject the gap is to feed the model an older observation.
That is the wrong way here, for a reason specific to this repository: the lidar
raster is cached and the cache is fingerprinted by its tick ages, so reading a
delayed sweep forces a full re-cache of the training set. The transform below
does the same job from the other side. Keep the observation at the anchor and
move the *label* to where the plan will be executed: re-anchor the future poses
onto the pose k ticks ahead, and take the planning horizon from there. A model
trained on that, shown the observation at t-k, outputs the plan that is correct
to apply at t -- in the frame it will be applied in.

Re-anchoring is what makes this exact rather than approximate. Shifting the
label window without changing frame would hand the model a plan expressed
around a pose it has already left, which is a constant offset it would learn to
add back: a systematic lag, precisely the artefact this is supposed to remove.

Frame freeze needs no separate training form. A frozen sensor at age a is an
observation delayed by a, and a burst of freeze is a sweep over ages; per sample
the two are the same transform, so a policy trained across the latency ages
already covers it. The difference between them is temporal correlation across
consecutive ticks, which a single-frame dataset cannot express in either case,
so freeze is evaluated rather than trained.
"""

import jaxtyping as jt
import numpy as np
import numpy.typing as npt


def shifted_planning_label(
    waypoints: jt.Float[npt.NDArray, "n 2"],
    yaws: jt.Float[npt.NDArray, " n"],
    shift_index: int,
    horizon: int,
) -> tuple[jt.Float[npt.NDArray, "h 2"], jt.Float[npt.NDArray, " h"]]:
    """Re-anchor a planning label onto a later pose and take the horizon there.

    Args:
        waypoints: Future positions in the anchor's frame, one per label tick,
            ordered by increasing tick.
        yaws: Future headings relative to the anchor's, same ordering.
        shift_index: How many label ticks ahead the plan is executed. Zero
            returns the leading ``horizon`` entries unchanged.
        horizon: Number of waypoints the model predicts.

    Returns:
        The re-anchored waypoints and yaws, both of length ``horizon``.

    Raises:
        ValueError: If the label does not reach far enough for this shift.
    """
    if shift_index < 0:
        raise ValueError(f"shift_index must not be negative, got {shift_index}.")
    if shift_index + horizon > len(waypoints):
        raise ValueError(
            f"A shift of {shift_index} label ticks needs {shift_index + horizon} "
            f"future poses but only {len(waypoints)} were read; raise "
            f"future_ego_pose_extra_ticks.",
        )
    if shift_index == 0:
        return waypoints[:horizon], yaws[:horizon]

    # The execution pose is itself one of the future poses, so no extra read is
    # needed to re-anchor onto it: it is the entry the shift lands on.
    origin = waypoints[shift_index - 1]
    heading = yaws[shift_index - 1]

    window = waypoints[shift_index : shift_index + horizon] - origin
    rotation = np.array(
        [
            [np.cos(-heading), -np.sin(-heading)],
            [np.sin(-heading), np.cos(-heading)],
        ],
        dtype=window.dtype,
    )
    shifted_waypoints = window @ rotation.T
    shifted_yaws = _normalize_angle(
        yaws[shift_index : shift_index + horizon] - heading,
    )
    return shifted_waypoints, shifted_yaws


def sample_shift_index(
    severity: float,
    max_shift_index: int,
    generator: np.random.Generator,
) -> int:
    """Draw how many label ticks of latency one sample carries.

    The draw is uniform over the reachable shifts rather than proportional to
    the severity, because the severity already decides the ceiling: a policy
    that only ever saw the worst delay would be tuned for it and mistimed
    everywhere else.

    Args:
        severity: Per-sample severity in ``[0, 1]``, scaling the ceiling.
        max_shift_index: Largest shift the read window supports.
        generator: Draws the shift.

    Returns:
        A shift in ``[0, max_shift_index]``.
    """
    ceiling = int(round(severity * max_shift_index))
    if ceiling <= 0:
        return 0
    return int(generator.integers(0, ceiling + 1))


def _normalize_angle(
    angle: jt.Float[npt.NDArray, " n"],
) -> jt.Float[npt.NDArray, " n"]:
    """Wrap angles into ``(-pi, pi]``.

    Args:
        angle: Angles in radians.

    Returns:
        The wrapped angles.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
