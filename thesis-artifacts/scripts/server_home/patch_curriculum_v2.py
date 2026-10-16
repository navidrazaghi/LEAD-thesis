# -*- coding: utf-8 -*-
"""Widen the degradation curriculum along the three axes the literature names.

The curriculum as written damages exactly one modality per sample, at a severity
drawn uniformly, and never touches the geometry between the two. Three gaps
follow from that, and each has a published counterpart:

  * No sample ever has both modalities damaged. Post-fusion BEV stabilization
    (arXiv 2603.05623) draws each modality independently at p = 0.30, which
    yields 49% clean / 21% camera / 21% LiDAR / 9% both; the 9% is the case a
    deployed stack meets in fog and this curriculum cannot produce at all. It is
    also the case this thesis lists as untested.
  * Total failure is vanishingly rare. Severity is uniform, so the fully-failed
    end -- the condition every closed-loop number in chapter 5 is scored at --
    is drawn with density zero. Grace-BEV (arXiv 2605.30983) trains with
    complete dropout at a fixed 25% + 25%, and CMT's masked-modal training is
    the same idea; MultiCorrupt (arXiv 2402.11677) reports that masked-modal
    training is what visibly buys robustness.
  * Nothing perturbs the camera/LiDAR geometry. MultiCorrupt finds spatial
    misalignment among the most damaging corruptions, and this stack is more
    exposed to it than most: the deformable operator's reference points are
    seeded from the nominal calibration, so a miscalibrated rig moves the world
    out from under a table the model treats as fixed.

All three are opt-in and default to off, because the module's own contract says
the default path must reproduce the established curriculum draw for draw -- the
rung that measured the curriculum's effect is defined by this sampler's random
stream, and a single extra draw on the default path would silently make it a
different experiment.
"""
import io
import pathlib
import sys

REPO = pathlib.Path.home() / "LEAD/lead"
DEG = REPO / "src/lead/policy/transfuser/utils/sensor_degradation.py"
CFG = REPO / "src/lead/config/training/data_config.py"
POLICY = REPO / "src/lead/policy/transfuser/transfuser.py"

MISALIGNMENT_FN = '''
# Extrinsic error at full severity: how far the rig's geometry is taken to be
# wrong. A metre of translation and three degrees of rotation are large for a
# calibrated vehicle and small next to the BEV extent, which is the range where
# the fusion has to keep working rather than fall over.
_MAX_MISALIGNMENT_METER = 1.0
_MAX_MISALIGNMENT_DEGREE = 3.0


def degrade_misalignment(
    rasterized_lidar: jt.Float[torch.Tensor, "b c h w"],
    severity: jt.Float[torch.Tensor, " b"],
    pixels_per_meter: float,
    generator: torch.Generator | None = None,
) -> jt.Float[torch.Tensor, "b c h w"]:
    """Move the BEV raster under the image, as a miscalibrated rig would.

    The two branches keep their own contents; only the geometry relating them
    is wrong, which is the failure a calibration drift produces and which no
    appearance family reproduces. The raster is rotated about its own centre
    and translated, with the amount scaled by the sample's severity and the
    direction drawn per sample.

    Args:
        rasterized_lidar: The collated BEV density raster.
        severity: Per-sample severity in ``[0, 1]``; zero leaves a sample alone.
        pixels_per_meter: Resolution of the raster, to turn metres into pixels.
        generator: Random source; None uses the global stream.

    Returns:
        The misaligned raster, same shape.
    """
    batch_size, _, height, width = rasterized_lidar.shape
    device, dtype = rasterized_lidar.device, rasterized_lidar.dtype
    amount = severity.to(device=device, dtype=torch.float32)

    angle = (
        (torch.rand(batch_size, generator=generator) * 2.0 - 1.0).to(device)
        * amount
        * math.radians(_MAX_MISALIGNMENT_DEGREE)
    )
    shift_meter = (
        torch.rand(batch_size, 2, generator=generator) * 2.0 - 1.0
    ).to(device) * amount[:, None] * _MAX_MISALIGNMENT_METER
    # affine_grid works in normalized coordinates, so metres become fractions
    # of each axis's half-extent.
    shift = shift_meter * pixels_per_meter * 2.0
    shift[:, 0] = shift[:, 0] / width
    shift[:, 1] = shift[:, 1] / height

    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
    theta[:, 1, 0], theta[:, 1, 1] = sin, cos
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(rasterized_lidar.shape), align_corners=False)
    moved = F.grid_sample(
        rasterized_lidar.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return moved.to(dtype)

'''

# --- the sampler: an independent-draw branch beside the legacy one ----------
OLD_SAMPLER = '''    batch_size = reference.shape[0]
    # Drawn on the host, like the colour augmentation, so no branch waits on
    # device-side randomness.
    selected = torch.rand(batch_size) < probability
    severity = torch.rand(batch_size) * max_severity * selected
    degrade_the_camera = torch.rand(batch_size) < 0.5

    if not deployment_families:'''

NEW_SAMPLER = '''    batch_size = reference.shape[0]
    # Drawn on the host, like the colour augmentation, so no branch waits on
    # device-side randomness.
    if independent_modalities:
        # Each modality draws for itself, so a sample can lose both. With
        # p = 0.3 this is the 49/21/21/9 split of arXiv 2603.05623.
        camera_hit = torch.rand(batch_size) < probability
        lidar_hit = torch.rand(batch_size) < probability
        camera_severity = torch.rand(batch_size) * max_severity * camera_hit
        lidar_severity = torch.rand(batch_size) * max_severity * lidar_hit
        if full_failure_probability > 0.0:
            # The condition every closed-loop number is scored at, drawn with
            # positive probability instead of uniform density zero.
            camera_out = (torch.rand(batch_size) < full_failure_probability) & camera_hit
            lidar_out = (torch.rand(batch_size) < full_failure_probability) & lidar_hit
            camera_severity = torch.where(camera_out, torch.full_like(camera_severity, max_severity), camera_severity)
            lidar_severity = torch.where(lidar_out, torch.full_like(lidar_severity, max_severity), lidar_severity)
        occlusion_severity = torch.zeros(batch_size)
        ego_severity = torch.zeros(batch_size)
        misalignment_severity = (
            torch.rand(batch_size)
            * max_severity
            * (torch.rand(batch_size) < misalignment_probability)
            if misalignment_probability > 0.0
            else torch.zeros(batch_size)
        )
        if "rasterized_lidar" in batch and bool((misalignment_severity > 0.0).any()):
            batch["rasterized_lidar"] = degrade_misalignment(
                batch["rasterized_lidar"],
                misalignment_severity,
                bev_pixels_per_meter,
            )
        return _apply_appearance(
            batch,
            camera_severity,
            lidar_severity,
            occlusion_severity,
            ego_severity,
        )

    selected = torch.rand(batch_size) < probability
    severity = torch.rand(batch_size) * max_severity * selected
    degrade_the_camera = torch.rand(batch_size) < 0.5

    if not deployment_families:'''

# --- the damage-and-retarget tail, shared by both branches -----------------
OLD_TAIL = '''    # Occlusion reports what it actually removed rather than what it was asked
    # to remove, so the target scale below is measured for this family and
    # posited for the others.
    camera_visible = torch.ones(batch_size)'''

NEW_TAIL = '''    return _apply_appearance(
        batch,
        camera_severity,
        lidar_severity,
        occlusion_severity,
        ego_severity,
    )


def _apply_appearance(
    batch: dict,
    camera_severity: jt.Float[torch.Tensor, " b"],
    lidar_severity: jt.Float[torch.Tensor, " b"],
    occlusion_severity: jt.Float[torch.Tensor, " b"],
    ego_severity: jt.Float[torch.Tensor, " b"],
) -> dict:
    """Damage the drawn modalities and scale the observability targets to match.

    Args:
        batch: The collated batch, modified in place.
        camera_severity: Per-sample camera severity.
        lidar_severity: Per-sample LiDAR severity.
        occlusion_severity: Per-sample occlusion severity.
        ego_severity: Per-sample ego-state severity.

    Returns:
        The same batch.
    """
    batch_size = camera_severity.shape[0]
    # Occlusion reports what it actually removed rather than what it was asked
    # to remove, so the target scale below is measured for this family and
    # posited for the others.
    camera_visible = torch.ones(batch_size)'''

SIGNATURE_OLD = '''def apply_sensor_degradation(
    batch: dict,
    probability: float,
    max_severity: float,
    deployment_families: Sequence[str] = (),
) -> dict:'''

SIGNATURE_NEW = '''def apply_sensor_degradation(
    batch: dict,
    probability: float,
    max_severity: float,
    deployment_families: Sequence[str] = (),
    independent_modalities: bool = False,
    full_failure_probability: float = 0.0,
    misalignment_probability: float = 0.0,
    bev_pixels_per_meter: float = 4.0,
) -> dict:'''

DOC_OLD = '''        deployment_families: Extra families to sample alongside the two
            appearance ones; ``"occlusion"`` and ``"ego_state"`` are
            implemented here, the temporal families are label transforms and
            live in the dataloader.'''

DOC_NEW = '''        deployment_families: Extra families to sample alongside the two
            appearance ones; ``"occlusion"`` and ``"ego_state"`` are
            implemented here, the temporal families are label transforms and
            live in the dataloader.
        independent_modalities: Draw the camera and the LiDAR separately, each
            at ``probability``, so a sample can lose both at once. Off keeps
            the exclusive draw, and with it the random stream of the runs that
            established the curriculum's effect.
        full_failure_probability: Chance that a modality already drawn for
            damage is taken to ``max_severity`` rather than a uniform draw,
            putting the fully-failed case -- the one the closed-loop protocol
            scores -- into training with positive probability. Independent
            draws only.
        misalignment_probability: Chance of moving the BEV raster under the
            image as a miscalibrated rig would, leaving both modalities intact
            but their geometry wrong. Independent draws only.
        bev_pixels_per_meter: Resolution of the BEV raster, for misalignment.'''

CONFIG_OLD = '''    sensor_degradation_max_severity: float = 1.0'''

CONFIG_NEW = '''    sensor_degradation_max_severity: float = 1.0
    # Curriculum v2, all off by default: see apply_sensor_degradation.
    sensor_degradation_independent_modalities: bool = False
    sensor_degradation_full_failure_probability: float = 0.0
    sensor_degradation_misalignment_probability: float = 0.0'''

CALLER_OLD = '''            batch = apply_sensor_degradation(
                batch,
                data_config.sensor_degradation_probability,
                data_config.sensor_degradation_max_severity,
                data_config.deployment_perturbation_families,
            )'''

CALLER_NEW = '''            batch = apply_sensor_degradation(
                batch,
                data_config.sensor_degradation_probability,
                data_config.sensor_degradation_max_severity,
                data_config.deployment_perturbation_families,
                data_config.sensor_degradation_independent_modalities,
                data_config.sensor_degradation_full_failure_probability,
                data_config.sensor_degradation_misalignment_probability,
                float(self.get_policy_config().bev_pixels_per_meter),
            )'''

IMPORT_OLD = '''from collections.abc import Sequence'''
IMPORT_NEW = '''import math
from collections.abc import Sequence'''

EDITS = [
    (DEG, IMPORT_OLD, IMPORT_NEW),
    (DEG, "def _gaussian_blur(", MISALIGNMENT_FN.lstrip("\n") + "\ndef _gaussian_blur("),
    (DEG, SIGNATURE_OLD, SIGNATURE_NEW),
    (DEG, DOC_OLD, DOC_NEW),
    (DEG, OLD_SAMPLER, NEW_SAMPLER),
    (DEG, OLD_TAIL, NEW_TAIL),
    (CFG, CONFIG_OLD, CONFIG_NEW),
    (POLICY, CALLER_OLD, CALLER_NEW),
]


def main() -> None:
    """Apply every edit, refusing anything ambiguous."""
    for path, anchor, replacement in EDITS:
        text = io.open(path, encoding="utf-8").read()
        found = text.count(anchor)
        if found != 1:
            sys.exit(f"FATAL: {path.name}: anchor appears {found} times, expected 1:\n{anchor[:120]}")
        io.open(path, "w", encoding="utf-8", newline="\n").write(text.replace(anchor, replacement))
        print(f"  patched {path.name}")


main()
