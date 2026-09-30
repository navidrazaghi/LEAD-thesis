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

from collections.abc import Sequence

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

# --- Deployment-perturbation families ---------------------------------------
#
# The three families above damage how a sensor *renders* the world: dimmer,
# blurrier, thinner. A deployed stack fails in ways that leave every rendering
# untouched -- a lens is blocked, the localisation drifts, the speedometer
# reads low. Those are different failure classes and the model has never seen
# them, which is exactly why they are worth training on.

# How many opaque patches an occluded image gets. Several small patches model a
# splash pattern or a partly blocked lens better than one large rectangle, which
# a convolutional encoder learns to recognise as a single object.
_NUM_OCCLUSION_PATCHES = 3
# Half-width of one patch at full severity, as a fraction of each image
# dimension. Three patches of this size cover about half the frame.
_MAX_OCCLUSION_HALF_FRACTION = 0.2
# Localisation jitter at full severity, in meters, applied to the navigation
# points the planner steers towards.
_MAX_GPS_JITTER_M = 3.0
# Fraction of true speed the measurement can lose at full severity. Speed is
# under-reported rather than jittered because that is the failure that matters:
# a policy that believes it is slower than it is accelerates into trouble.
_MAX_SPEED_UNDERESTIMATE = 0.35

# The deployment families this module applies to a collated batch. Latency and
# frame freeze are absent on purpose: delaying an observation means reading a
# different tick, which is a dataloader concern, and the training-time form of
# both is a shift of the planning label rather than a change to any input.
_BATCH_LEVEL_FAMILIES = ("occlusion", "ego_state")


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
    rgb: jt.Real[torch.Tensor, "b 3 h w"],
    severity: jt.Float[torch.Tensor, " b"],
    generator: torch.Generator | None = None,
) -> jt.Real[torch.Tensor, "b 3 h w"]:
    """Dim, blur and add noise to each image in proportion to its severity.

    The two callers hand this different dtypes for the same pixels. Training
    degrades the collated uint8 batch; at inference ``features_to_batch`` has
    already cast every model input to float32. Both carry 0-255 values -- the
    backbone divides by 255 itself -- so the arithmetic below is identical and
    only the annotation has to admit both. Narrowing it to uint8 makes every
    degraded evaluation run raise under the default runtime type checking.

    Args:
        rgb: The collated camera batch, 0-255 valued, uint8 or float.
        severity: Per-sample severity in ``[0, 1]``; zero leaves a sample alone.
        generator: Draws the noise, so a run can repeat its own damage. None
            uses the global stream, which is what training wants.

    Returns:
        The degraded batch, same dtype, device and layout.
    """
    x = rgb.to(torch.float32)
    per_sample = severity.view(-1, 1, 1, 1).to(x.device, x.dtype)

    # One blur for the whole batch, then mixed in per sample: a per-sample sigma
    # would need a kernel each, and the mix already spans no blur to full blur.
    # The cast back is load-bearing: the blur is a conv2d, so under autocast it
    # returns bfloat16 while x is still float32, and lerp refuses the mismatch.
    blurred = _gaussian_blur(x, _MAX_BLUR_SIGMA).to(x.dtype)
    x = torch.lerp(x, blurred, per_sample)

    gain = 1.0 - (1.0 - _MIN_IMAGE_GAIN) * per_sample
    x = x * gain

    noise = torch.randn(
        x.shape,
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    ) * (_MAX_NOISE_STD * 255.0) * per_sample
    return (x + noise).clamp(0.0, 255.0).to(rgb.dtype)


def degrade_lidar(
    rasterized_lidar: jt.Float[torch.Tensor, "b c h w"],
    severity: jt.Float[torch.Tensor, " b"],
    generator: torch.Generator | None = None,
) -> jt.Float[torch.Tensor, "b c h w"]:
    """Drop returns from the BEV raster in proportion to each sample's severity.

    Thinning the splat is what range, rain and scattering do to a sweep: cells
    lose returns, they do not gain wrong ones.

    Args:
        rasterized_lidar: The collated BEV density raster.
        severity: Per-sample severity in ``[0, 1]``.
        generator: Draws the dropout, so a run can repeat its own damage. None
            uses the global stream, which is what training wants.

    Returns:
        The thinned raster, same shape and dtype.
    """
    per_sample = severity.view(-1, 1, 1, 1).to(
        rasterized_lidar.device,
        rasterized_lidar.dtype,
    )
    draw = torch.rand(
        rasterized_lidar.shape,
        generator=generator,
        device=rasterized_lidar.device,
        dtype=rasterized_lidar.dtype,
    )
    keep = draw >= (_MAX_POINT_DROPOUT * per_sample)
    return rasterized_lidar * keep


def degrade_occlusion(
    rgb: jt.Real[torch.Tensor, "b 3 h w"],
    severity: jt.Float[torch.Tensor, " b"],
    generator: torch.Generator | None = None,
) -> tuple[jt.Real[torch.Tensor, "b 3 h w"], jt.Float[torch.Tensor, " b"]]:
    """Black out rectangular regions of each image in proportion to its severity.

    This is the one appearance failure that leaves the rest of the frame
    pristine: everything outside the patch is exactly as sharp and bright as it
    was, so the encoder cannot detect the fault from global statistics the way
    it can with dimming, blur or noise. That is the point -- a blocked lens is
    locally total and globally invisible.

    Args:
        rgb: The collated camera batch, 0-255 valued, uint8 or float.
        severity: Per-sample severity in ``[0, 1]``; zero leaves a sample alone.
        generator: Draws the patch placement, so a run can repeat its own
            damage. None uses the global stream, which is what training wants.

    Returns:
        The occluded batch in the input dtype, and the fraction of each image
        that survived. The second value is what the observability targets are
        scaled by: unlike the other families, how much this damage costs is
        measurable from the mask rather than posited from the severity.
    """
    batch_size, _, height, width = rgb.shape
    device = rgb.device
    per_sample = severity.to(device, torch.float32)

    rows = torch.arange(height, device=device, dtype=torch.float32)
    columns = torch.arange(width, device=device, dtype=torch.float32)

    occluded = torch.zeros(batch_size, 1, height, width, device=device, dtype=torch.bool)
    for _ in range(_NUM_OCCLUSION_PATCHES):
        centre_row = torch.rand(
            batch_size, generator=generator, device=device,
        ) * height
        centre_column = torch.rand(
            batch_size, generator=generator, device=device,
        ) * width
        half_row = per_sample * _MAX_OCCLUSION_HALF_FRACTION * height
        half_column = per_sample * _MAX_OCCLUSION_HALF_FRACTION * width
        # A zero-severity sample gets a zero-extent patch, so the strict
        # comparison leaves it untouched without a branch.
        inside_rows = (rows.view(1, -1) - centre_row.view(-1, 1)).abs() < half_row.view(-1, 1)
        inside_columns = (
            columns.view(1, -1) - centre_column.view(-1, 1)
        ).abs() < half_column.view(-1, 1)
        patch = inside_rows.view(batch_size, 1, height, 1) & inside_columns.view(
            batch_size, 1, 1, width,
        )
        occluded = occluded | patch

    visible_fraction = 1.0 - occluded.to(torch.float32).mean(dim=(1, 2, 3))
    return rgb * (~occluded).to(rgb.dtype), visible_fraction


def degrade_ego_state(
    batch: dict,
    severity: jt.Float[torch.Tensor, " b"],
    generator: torch.Generator | None = None,
) -> dict:
    """Jitter the navigation points and under-report the speed, per sample.

    This family is deliberately outside the modality split that governs the
    rest of this module. A drifting GPS fix or a slow speedometer does not make
    either sensor resolve the scene any less well, so nothing here scales the
    observability targets -- doing so would teach the gate to shift modality on
    a fault that has no modality. What it does damage is the ego state the
    planner conditions on, an input this stack had never perturbed before.

    Args:
        batch: The collated batch, modified in place.
        severity: Per-sample severity in ``[0, 1]``; zero leaves a sample alone.
        generator: Draws the jitter, so a run can repeat its own damage. None
            uses the global stream, which is what training wants.

    Returns:
        The same batch.
    """
    speed = batch.get("speed")
    if speed is not None:
        lost = severity.to(speed.device, speed.dtype) * _MAX_SPEED_UNDERESTIMATE
        batch["speed"] = speed * (1.0 - lost)

    for key in ("previous_target_point", "target_point", "next_target_point"):
        points = batch.get(key)
        if points is None:
            continue
        jitter = torch.randn(
            points.shape,
            generator=generator,
            device=points.device,
            dtype=points.dtype,
        ) * _MAX_GPS_JITTER_M
        batch[key] = points + jitter * severity.to(points.device, points.dtype).view(-1, 1)

    return batch


def apply_sensor_degradation(
    batch: dict,
    probability: float,
    max_severity: float,
    deployment_families: Sequence[str] = (),
) -> dict:
    """Degrade one family of some samples, and scale their targets to match.

    Only one family is applied per sample. For the two appearance families that
    is the original reason: the point of the gate is to shift reading onto
    whichever modality is still intact, and damaging both at once would leave it
    nowhere to shift to. For the deployment families it is the same discipline
    for a different reason -- stacking an occlusion on top of a speed fault
    would leave no way to attribute what the model learned to either.

    ``deployment_families`` empty reproduces the appearance-only curriculum
    exactly, draw for draw. That matters more than it looks: the rung that
    established the curriculum's effect is defined by this sampler's random
    stream, so any extra draw taken on the default path would silently make it
    a different experiment.

    Args:
        batch: The collated batch, modified in place.
        probability: Chance a sample is degraded at all.
        max_severity: Upper bound of the sampled severity.
        deployment_families: Extra families to sample alongside the two
            appearance ones; ``"occlusion"`` and ``"ego_state"`` are
            implemented here, the temporal families are label transforms and
            live in the dataloader.

    Returns:
        The same batch.

    Raises:
        ValueError: If a requested family is not one this function applies.
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

    if not deployment_families:
        camera_severity = severity * degrade_the_camera
        lidar_severity = severity * ~degrade_the_camera
        occlusion_severity = torch.zeros(batch_size)
        ego_severity = torch.zeros(batch_size)
    else:
        unknown = set(deployment_families) - set(_BATCH_LEVEL_FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown deployment perturbation families {sorted(unknown)}; "
                f"this function applies {sorted(_BATCH_LEVEL_FAMILIES)}.",
            )
        # The appearance families keep their shared draw so that the camera and
        # lidar arms stay balanced against each other, and the deployment ones
        # are drawn as additional equally likely alternatives.
        families = ("appearance", *deployment_families)
        chosen = torch.randint(len(families), (batch_size,))
        is_appearance = chosen == 0
        camera_severity = severity * is_appearance * degrade_the_camera
        lidar_severity = severity * is_appearance * ~degrade_the_camera
        occlusion_severity = severity * (
            chosen == families.index("occlusion") if "occlusion" in families
            else torch.zeros(batch_size, dtype=torch.bool)
        )
        ego_severity = severity * (
            chosen == families.index("ego_state") if "ego_state" in families
            else torch.zeros(batch_size, dtype=torch.bool)
        )

    # Occlusion reports what it actually removed rather than what it was asked
    # to remove, so the target scale below is measured for this family and
    # posited for the others.
    camera_visible = torch.ones(batch_size)
    if "rgb" in batch:
        batch["rgb"] = degrade_camera(batch["rgb"], camera_severity)
        if bool((occlusion_severity > 0.0).any()):
            batch["rgb"], camera_visible = degrade_occlusion(
                batch["rgb"],
                occlusion_severity,
            )
            camera_visible = camera_visible.to(torch.float32).cpu()
    if "rasterized_lidar" in batch:
        batch["rasterized_lidar"] = degrade_lidar(
            batch["rasterized_lidar"],
            lidar_severity,
        )
    if bool((ego_severity > 0.0).any()):
        batch = degrade_ego_state(batch, ego_severity)

    if "observability" in batch:
        target = batch["observability"]
        survives = torch.stack(
            [(1.0 - camera_severity) * camera_visible, 1.0 - lidar_severity],
            dim=1,
        ).to(target.device, target.dtype)
        # Channel order is the enum's, so the scale lands on the modality that
        # was actually damaged.
        assert survives.shape[1] == len(ObservabilityChannel)
        batch["observability"] = target * survives[:, :, None, None]

    return batch


def degrade_batch_family(
    batch: dict,
    family: str,
    severity: float,
    generator: torch.Generator | None = None,
) -> dict:
    """Apply one deployment family to a whole inference batch at a fixed severity.

    The deployment families were reachable only from the training curriculum,
    which draws them per sample. Evaluation needs the opposite: the same fault
    on every sample, so a run is one point on a curve rather than an average
    over random ones -- the same reason :func:`degrade_batch` exists beside the
    curriculum for the appearance families.

    Without this, a rung trained on occlusion and ego-state noise could only be
    scored under camera and lidar destruction, which answers whether the extra
    families cost anything elsewhere and not whether they bought anything where
    they were aimed.

    Args:
        batch: The collated model inputs, modified in place.
        family: ``"occlusion"``, ``"ego_state"``, or ``"none"``.
        severity: How much to damage it, in ``[0, 1]``, applied to every sample.
        generator: Draws the damage, so a run can repeat itself.

    Returns:
        The batch.

    Raises:
        ValueError: If the family is not one this applies. Silently ignoring an
            unknown name would score an undamaged run under a condition's label.
    """
    if family == "none" or severity <= 0.0:
        return batch
    if family not in _BATCH_LEVEL_FAMILIES:
        raise ValueError(
            f"unknown deployment family '{family}'; this applies "
            f"{sorted(_BATCH_LEVEL_FAMILIES)}.",
        )

    reference = batch.get("rgb")
    if reference is None:
        reference = batch.get("rasterized_lidar")
    if reference is None:
        return batch
    per_sample = torch.full(
        (reference.shape[0],),
        float(severity),
        device=reference.device,
        dtype=torch.float32,
    )

    if family == "occlusion":
        if "rgb" not in batch:
            return batch
        batch["rgb"], _ = degrade_occlusion(batch["rgb"], per_sample, generator)
        return batch
    return degrade_ego_state(batch, per_sample, generator)


def degrade_batch(
    batch: dict,
    modality: str,
    severity: float,
    generator: torch.Generator | None = None,
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
        generator: Draws the damage. Seeding it is what lets two checkpoints be
            compared on the same route under the *same* damage rather than two
            samples from the same distribution.

    Returns:
        The same batch.

    Raises:
        ValueError: If ``modality`` is not one of the three accepted names.
    """
    if modality == "none" or severity <= 0.0:
        return batch
    accepted = ("camera", "lidar", "both", *_BATCH_LEVEL_FAMILIES)
    if modality not in accepted:
        raise ValueError(
            f"degrade modality must be one of {(*accepted, 'none')}, "
            f"got '{modality}'.",
        )

    reference = batch.get("rgb")
    if reference is None:
        reference = batch.get("rasterized_lidar")
    if reference is None:
        return batch

    amount = torch.full((reference.shape[0],), float(severity))
    if modality == "camera" and "rgb" in batch:
        batch["rgb"] = degrade_camera(batch["rgb"], amount, generator)
    elif modality == "lidar" and "rasterized_lidar" in batch:
        batch["rasterized_lidar"] = degrade_lidar(
            batch["rasterized_lidar"],
            amount,
            generator,
        )
    elif modality == "occlusion" and "rgb" in batch:
        # The visible fraction is the training signal's, not the driver's, so
        # inference keeps only the damaged image.
        batch["rgb"], _ = degrade_occlusion(batch["rgb"], amount, generator)
    elif modality == "ego_state":
        batch = degrade_ego_state(batch, amount, generator)
    elif modality == "both":
        # The one regime redundancy cannot cover, and the only one where an
        # estimate of what is still resolved has anything to say: with one
        # modality down the other carries the scene, which is measurable here --
        # the trained rungs score no worse under full camera destruction than
        # intact. Training never produces this, by design, because the gate it
        # was built for needs somewhere to shift to. It is an evaluation
        # condition, and a deliberately out-of-family one.
        if "rgb" in batch:
            batch["rgb"] = degrade_camera(batch["rgb"], amount, generator)
        if "rasterized_lidar" in batch:
            batch["rasterized_lidar"] = degrade_lidar(
                batch["rasterized_lidar"],
                amount,
                generator,
            )
    return batch
