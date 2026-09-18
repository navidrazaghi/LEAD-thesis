"""Degrade one modality per sample, and say so in the observability targets.

The dataset is recorded under one set of conditions, so nothing in it teaches a
model what a failing camera looks like. Without that, an observability head only
ever learns occlusion — which actor is hidden behind which — and a gate trained
on it has no reason to react when a sensor itself goes bad.

This applies the missing variation directly: pick a modality, damage it by a
sampled severity, and scale that modality's observability targets by what
survives. The pairing is the point. Damaging the input alone would train the
head to insist the camera still sees everything; scaling the target alone would
train it to cry wolf. Together they say: this is what a degraded camera looks
like, and this is how much less it now resolves.

Scaling the target linearly in severity is a modelling assumption, not a
measurement — the dataset records visibility under nominal sensors only, so the
degraded value cannot be observed, only posited.
"""

import jaxtyping as jt
import torch
from torch.nn import functional as F

from lead.policy.transfuser.dataloader.observability import ObservabilityChannel

# Severity 1 leaves this much of the image: not fully black, so the encoder
# still sees an input rather than a constant it can special-case.
_MIN_IMAGE_GAIN = 0.05
# Blur kernel at full severity, in pixels of the stitched image.
_MAX_BLUR_SIGMA = 6.0
# Additive noise at full severity, as a fraction of full scale.
_MAX_NOISE_STD = 0.25
# Fraction of occupied BEV cells dropped at full severity.
_MAX_POINT_DROPOUT = 0.95


def _gaussian_blur(
    x: jt.Float[torch.Tensor, "b c h w"],
    sigma: float,
) -> jt.Float[torch.Tensor, "b c h w"]:
    """Blur a batch with one separable Gaussian kernel.

    Args:
        x: The batch to blur, channel first.
        sigma: Standard deviation of the kernel in pixels.

    Returns:
        The blurred batch, same shape.
    """
    radius = max(int(3.0 * sigma), 1)
    taps = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(x, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(x, vertical, padding=(radius, 0), groups=channels)


def degrade_camera(
    rgb: jt.UInt8[torch.Tensor, "b 3 h w"],
    severity: jt.Float[torch.Tensor, " b"],
) -> jt.UInt8[torch.Tensor, "b 3 h w"]:
    """Dim, blur and add noise to each image in proportion to its severity.

    Args:
        rgb: The collated camera batch.
        severity: Per-sample severity in ``[0, 1]``; zero leaves a sample alone.

    Returns:
        The degraded batch, same dtype, device and layout.
    """
    x = rgb.to(torch.float32)
    per_sample = severity.view(-1, 1, 1, 1).to(x.device, x.dtype)

    # One blur for the whole batch, then mixed in per sample: a per-sample sigma
    # would need a kernel each, and the mix already spans no blur to full blur.
    blurred = _gaussian_blur(x, _MAX_BLUR_SIGMA)
    x = torch.lerp(x, blurred, per_sample)

    gain = 1.0 - (1.0 - _MIN_IMAGE_GAIN) * per_sample
    x = x * gain

    noise = torch.randn_like(x) * (_MAX_NOISE_STD * 255.0) * per_sample
    return (x + noise).clamp(0.0, 255.0).to(rgb.dtype)


def degrade_lidar(
    rasterized_lidar: jt.Float[torch.Tensor, "b c h w"],
    severity: jt.Float[torch.Tensor, " b"],
) -> jt.Float[torch.Tensor, "b c h w"]:
    """Drop returns from the BEV raster in proportion to each sample's severity.

    Thinning the splat is what range, rain and scattering do to a sweep: cells
    lose returns, they do not gain wrong ones.

    Args:
        rasterized_lidar: The collated BEV density raster.
        severity: Per-sample severity in ``[0, 1]``.

    Returns:
        The thinned raster, same shape and dtype.
    """
    per_sample = severity.view(-1, 1, 1, 1).to(
        rasterized_lidar.device,
        rasterized_lidar.dtype,
    )
    keep = torch.rand_like(rasterized_lidar) >= (_MAX_POINT_DROPOUT * per_sample)
    return rasterized_lidar * keep


def apply_sensor_degradation(
    batch: dict,
    probability: float,
    max_severity: float,
) -> dict:
    """Degrade one modality of some samples, and scale their targets to match.

    Only one modality is degraded per sample: the point of the gate is to shift
    reading onto whichever modality is still intact, and damaging both at once
    would leave it nowhere to shift to.

    Args:
        batch: The collated batch, modified in place.
        probability: Chance a sample is degraded at all.
        max_severity: Upper bound of the sampled severity.

    Returns:
        The same batch.
    """
    reference = batch.get("rgb")
    if reference is None:
        reference = batch.get("rasterized_lidar")
    if reference is None:
        return batch

    batch_size = reference.shape[0]
    # Drawn on the host, like the colour augmentation, so no branch waits on
    # device-side randomness.
    selected = torch.rand(batch_size) < probability
    severity = torch.rand(batch_size) * max_severity * selected
    degrade_the_camera = torch.rand(batch_size) < 0.5

    camera_severity = severity * degrade_the_camera
    lidar_severity = severity * ~degrade_the_camera

    if "rgb" in batch:
        batch["rgb"] = degrade_camera(batch["rgb"], camera_severity)
    if "rasterized_lidar" in batch:
        batch["rasterized_lidar"] = degrade_lidar(
            batch["rasterized_lidar"],
            lidar_severity,
        )

    if "observability" in batch:
        target = batch["observability"]
        survives = torch.stack(
            [1.0 - camera_severity, 1.0 - lidar_severity],
            dim=1,
        ).to(target.device, target.dtype)
        # Channel order is the enum's, so the scale lands on the modality that
        # was actually damaged.
        assert survives.shape[1] == len(ObservabilityChannel)
        batch["observability"] = target * survives[:, :, None, None]

    return batch


def degrade_batch(
    batch: dict,
    modality: str,
    severity: float,
) -> dict:
    """Damage one modality of an inference batch by a fixed amount.

    The training curriculum draws a modality and a severity per sample, which
    is what teaches the gate; measuring what the gate learned needs the
    opposite — the same damage applied to every sample, so a run is one point
    on a degradation curve rather than an average over random ones. The damage
    itself is the training curriculum's, so the two stay comparable.

    Args:
        batch: The collated model inputs, modified in place.
        modality: ``"camera"``, ``"lidar"``, or ``"none"`` to leave it alone.
        severity: How much to damage it, in ``[0, 1]``.

    Returns:
        The same batch.

    Raises:
        ValueError: If ``modality`` is not one of the three accepted names.
    """
    if modality == "none" or severity <= 0.0:
        return batch
    if modality not in ("camera", "lidar"):
        raise ValueError(
            f"degrade modality must be 'camera', 'lidar' or 'none', got '{modality}'.",
        )

    reference = batch.get("rgb")
    if reference is None:
        reference = batch.get("rasterized_lidar")
    if reference is None:
        return batch

    amount = torch.full((reference.shape[0],), float(severity))
    if modality == "camera" and "rgb" in batch:
        batch["rgb"] = degrade_camera(batch["rgb"], amount)
    elif modality == "lidar" and "rasterized_lidar" in batch:
        batch["rasterized_lidar"] = degrade_lidar(batch["rasterized_lidar"], amount)
    return batch
