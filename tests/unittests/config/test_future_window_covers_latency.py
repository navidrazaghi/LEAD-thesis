"""The scene filter has to demand the ticks a shifted label is re-anchored onto.

`future_ego_pose_extra_ticks` widens which future poses are read so a planning
label can be moved onto the pose the plan will actually be executed at. The
filter that decides how much future a scene must supply ignored those ticks, in
two places: the base policy config and the TransFuser override that shadows it.

The consequence was not a quiet degradation. The loader asks for the later
iterations, the filter drops every one past its own number, and
`shifted_planning_label` raises because it was handed an array shorter than the
shift plus the horizon -- so the latency curriculum could not have run at all.
It was never turned on, so nobody found out.

These tests pin both properties, and pin the one that must *not* include the
extra ticks, because the two mistakes are opposite and equally silent: widening
what is predicted would change the head's output width and orphan every
checkpoint.
"""

import pytest

from lead.config import load_lead_config


@pytest.fixture
def transfuser_config():
    """The TransFuser policy config, with planning labels built.

    Returns:
        The policy config node.
    """
    config = load_lead_config()
    config.policy.transfuser.use_planning_decoder = True
    return config.policy.transfuser


@pytest.mark.parametrize("extra", [0, 5, 10, 15])
def test_filter_demands_the_extra_ticks(transfuser_config, extra: int) -> None:
    """The scene filter must cover the whole read window.

    Args:
        transfuser_config: The policy config.
        extra: Latency ticks to configure.
    """
    transfuser_config.future_ego_pose_extra_ticks = extra
    demanded = transfuser_config.future_window_num_iterations
    read_to = transfuser_config.future_ego_pose_iterations[-1]
    assert demanded >= read_to, (
        f"the filter demands {demanded} ticks while the loader reads to tick "
        f"{read_to}; the difference is dropped and the latency shift then "
        f"raises on an array it cannot use."
    )


@pytest.mark.parametrize("extra", [0, 5, 10, 15])
def test_prediction_width_ignores_the_extra_ticks(
    transfuser_config,
    extra: int,
) -> None:
    """The head's width must not move with the read window.

    Args:
        transfuser_config: The policy config.
        extra: Latency ticks to configure.
    """
    transfuser_config.future_ego_pose_extra_ticks = 0
    baseline = transfuser_config.num_ego_pose_prediction
    transfuser_config.future_ego_pose_extra_ticks = extra
    assert transfuser_config.num_ego_pose_prediction == baseline, (
        "turning the latency curriculum on changed how many waypoints the "
        "policy predicts, which would make every existing checkpoint "
        "incompatible with it."
    )


def test_a_shift_has_poses_to_land_on(transfuser_config) -> None:
    """Read labels must outnumber the horizon by at least the maximum shift.

    This is the arithmetic `shifted_planning_label` enforces at runtime, checked
    here against the config instead so a misconfiguration is caught before a
    training run rather than a few hundred steps into one.

    Args:
        transfuser_config: The policy config.
    """
    transfuser_config.future_ego_pose_extra_ticks = 10
    stride = transfuser_config.future_ego_pose_iterations[0]
    max_shift = transfuser_config.future_ego_pose_extra_ticks // stride
    labels_read = len(transfuser_config.future_ego_pose_iterations)
    horizon = transfuser_config.num_ego_pose_prediction
    assert max_shift + horizon <= labels_read, (
        f"a shift of {max_shift} onto a {horizon}-step horizon needs "
        f"{max_shift + horizon} labels and only {labels_read} are read."
    )
