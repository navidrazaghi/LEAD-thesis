"""Deployment families have to be applicable at inference, not only in training.

rung2d was trained on occlusion and ego-state noise and could only have been
scored under camera and lidar destruction, because the inference config exposed
a modality and a severity and nothing else. That evaluation answers whether the
extra families cost anything elsewhere; it cannot answer whether they bought
anything where they were aimed.

These tests pin the inference-side application: that it damages, that it is a
no-op when asked for nothing, and that an unknown name raises instead of
quietly producing an undamaged run under a condition's label -- which would be
the worst outcome available, a clean number for an experiment that never ran.
"""

import pytest
import torch

from lead.policy.transfuser.utils.sensor_degradation import degrade_batch_family


def _batch(batch_size: int = 4):
    """A minimal collated batch with the fields the families touch.

    Args:
        batch_size: How many samples.

    Returns:
        The batch.
    """
    return {
        "rgb": torch.full((batch_size, 3, 32, 64), 128.0),
        "ego_speed": torch.full((batch_size, 1), 5.0),
        "target_point": torch.zeros(batch_size, 2),
    }


def test_occlusion_blacks_out_part_of_the_image() -> None:
    """Occlusion at full severity must change the camera batch."""
    batch = _batch()
    before = batch["rgb"].clone()
    generator = torch.Generator().manual_seed(0)
    after = degrade_batch_family(batch, "occlusion", 1.0, generator)["rgb"]
    assert not torch.equal(before, after)
    assert after.min() < before.min() or (after == 0).any()


def test_ego_state_leaves_the_image_alone() -> None:
    """The ego-state family must not touch the camera.

    It is deliberately outside the modality split: a drifting fix does not make
    either sensor resolve the scene less well, and damaging the image here
    would blur exactly the distinction the family exists to draw.
    """
    batch = _batch()
    before = batch["rgb"].clone()
    generator = torch.Generator().manual_seed(0)
    after = degrade_batch_family(batch, "ego_state", 1.0, generator)
    assert torch.equal(before, after["rgb"])


@pytest.mark.parametrize(
    ("family", "severity"),
    [("none", 1.0), ("occlusion", 0.0), ("ego_state", 0.0)],
)
def test_nothing_asked_nothing_damaged(family: str, severity: float) -> None:
    """A family of none, or a severity of zero, must leave the batch identical.

    Args:
        family: The family to request.
        severity: The severity to request.
    """
    batch = _batch()
    before = batch["rgb"].clone()
    after = degrade_batch_family(batch, family, severity)
    assert torch.equal(before, after["rgb"])


def test_an_unknown_family_raises() -> None:
    """An unrecognised name must not pass silently.

    Ignoring it would score an undamaged run and file the result under the
    condition's label, which looks like a measurement and is not one.
    """
    with pytest.raises(ValueError, match="unknown deployment family"):
        degrade_batch_family(_batch(), "fog", 1.0)


def test_severity_scales_the_damage() -> None:
    """More severity must not damage less.

    A monotone check rather than an exact one: the patch geometry is random, so
    what can be pinned is the direction.
    """
    changed = []
    for severity in (0.25, 1.0):
        batch = _batch(batch_size=16)
        before = batch["rgb"].clone()
        generator = torch.Generator().manual_seed(0)
        after = degrade_batch_family(batch, "occlusion", severity, generator)["rgb"]
        changed.append((before != after).float().mean().item())
    assert changed[1] >= changed[0]
