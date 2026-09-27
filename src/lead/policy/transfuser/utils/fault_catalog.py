"""A catalogue of structured camera and LiDAR faults, for evaluation first.

The robustness numbers in this repository measure one failure per sensor: the
camera dimmed, blurred and drowned in noise across the whole frame, the LiDAR
raster thinned uniformly by up to 95%. Those answer whether a policy can drive
when a sensor is effectively gone. They say nothing about the faults a deployed
stack meets more often -- mud on one lens, a flare, night, a LiDAR that loses
its far returns in rain or goes blind in one direction -- which are local,
structured by range and angle, and sometimes add returns rather than remove
them.

The kinds of fault here follow the catalogue of VG-SAF (Tao et al., "Variance-
Guided Spatial Attention Fusion for Robust End-to-End Driving under Asymmetric
Sensor Degradation", arXiv 2608.24366, Tables S1 and S2): eight camera modes and
five LiDAR modes. Their code is not released, so every functional form and
constant below is ours; only the list of phenomena is theirs. One thing is
deliberately different: VG-SAF evaluates on the same modes it trains on. Here
the catalogue is an evaluation instrument, and any training use is meant to
hold modes out, so that a gain measured on a mode is not a gain from having
seen it.

Two properties matter for evaluation and are handled explicitly:

* **Persistence.** Mud stays where it landed and a blocked LiDAR wedge does not
  rotate every tick. The geometry of a fault -- where the occluders are, which
  camera they are on, which way the wedge points -- is drawn from a generator
  re-seeded from ``persistent_seed`` on every call, so it is identical on every
  tick of a route. What genuinely varies in time -- sensor noise, speckle,
  dropped returns -- is drawn from the advancing per-route ``generator``.
  In training ``persistent_seed`` is None and the geometry is fresh per sample.
* **Sensor boundaries.** The model's image is its input cameras stitched side
  by side, so anything spatial (blur, ghosting, lens faults) is applied per
  camera tile: a convolution across the seam would smear one camera into the
  next, and a lens fault belongs to one lens.

Severity is in ``[0, 1]`` for every mode, zero leaving the input untouched. It
is a strength, not VG-SAF's mask-mean K, so severities of different modes are
not directly comparable; each mode's range is chosen so that 1.0 is the worst
version of that phenomenon, not the destruction of the sensor.

Camera inputs are 0-255 valued (uint8 in training, float at inference) and the
LiDAR input is the normalised BEV point-density raster in ``[0, 1]``, rows
along the lateral axis and columns along the longitudinal one, as
``features.rasterize_lidar_bev`` builds it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

CAMERA_FAULTS = (
    "cam_signal_drop",
    "cam_local_noise",
    "cam_exposure_pulse",
    "cam_local_occlusion",
    "cam_night_lowlight",
    "cam_motion_blur",
    "cam_ghosting",
    "cam_color_shift",
)
LIDAR_FAULTS = (
    "lid_signal_drop",
    "lid_range_dropout",
    "lid_frustum_occlusion",
    "lid_local_speckle",
    "lid_feature_noise",
)
FAULT_CATALOG = CAMERA_FAULTS + LIDAR_FAULTS

# Range at which a LiDAR return counts as fully "far" for range-conditioned
# effects. The raster's farthest corner is about 75 m from the sensor.
_FAR_RANGE_M = 70.0


@dataclass(frozen=True)
class FaultGeometry:
    """What the faults need to know about the input layout.

    Defaults are LEAD's: three stitched cameras and a 4 px/m BEV raster spanning
    -32..64 m longitudinally and -40..40 m laterally.
    """

    num_cameras: int = 3
    bev_pixels_per_meter: float = 4.0
    bev_min_x_meter: float = -32.0
    bev_min_y_meter: float = -40.0


# --- random draws ------------------------------------------------------------


def _draw(shape, generator, device) -> torch.Tensor:
    """Uniform draws from the advancing generator: what varies tick to tick."""
    return torch.rand(shape, generator=generator, device=device)


def _draw_normal(shape, generator, device) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device)


def _persistent(persistent_seed: int | None, generator, device, salt: int):
    """A draw function for fault geometry that must not change between ticks.

    With a seed, a CPU generator is rebuilt from it on every call, so the same
    route draws the same geometry every tick. ``salt`` keeps two modes from
    sharing their draws. Without a seed (training), geometry comes from the
    ordinary generator and is fresh per sample.
    """
    if persistent_seed is None:
        return lambda shape: _draw(shape, generator, device)
    local = torch.Generator(device="cpu")
    local.manual_seed((int(persistent_seed) * 1_000_003 + salt) % (2**63 - 1))
    return lambda shape: torch.rand(shape, generator=local).to(device)


# --- camera helpers ----------------------------------------------------------


def _tiles(width: int, num_cameras: int) -> list[tuple[int, int]]:
    """Column ranges of the stitched cameras."""
    tile = width // num_cameras
    return [(k * tile, (k + 1) * tile if k < num_cameras - 1 else width) for k in range(num_cameras)]


def _grid(height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
    return rows, cols


def _luminance(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


# --- camera faults -----------------------------------------------------------


def _cam_signal_drop(x, s, generator, draw_p, geometry):
    """Power loss, glare or snow: the frame goes black, white, or to static.

    Which of the three is fixed per route; at full severity the image is the
    fault and nothing of the scene remains, which is VG-SAF's 100% mode.
    """
    b = x.shape[0]
    variant = (draw_p((b,)) * 3).floor().clamp(max=2).view(b, 1, 1, 1)
    black = torch.zeros_like(x)
    white = torch.full_like(x, 255.0)
    # Snow: bright static correlated with the scene's own luminance, with a
    # small chromatic jitter so it is not perfectly grey.
    static = _draw(x.shape[:1] + (1,) + x.shape[2:], generator, x.device) * 255.0
    snow = (0.6 * static + 0.4 * _luminance(x)).expand_as(x)
    snow = snow * (1.0 + 0.08 * _draw_normal((b, 3, 1, 1), generator, x.device))
    target = torch.where(variant == 0, black, torch.where(variant == 1, white, snow))
    k = s.view(b, 1, 1, 1)
    return (1.0 - k) * x + k * target


def _soft_ellipses(b, height, width, count, centre_r, centre_c, half_r, half_c, enabled, device):
    """Union of soft-edged ellipses as a mask in [0, 1], shape (b, 1, h, w).

    ``enabled`` (b, count) switches ellipses off by multiplication; shrinking a
    disabled one to zero size would still leave a one-pixel dot at its centre.
    """
    rows, cols = _grid(height, width, device)
    mask = torch.zeros(b, 1, height, width, device=device)
    for i in range(count):
        d = (((rows - centre_r[:, i].view(-1, 1, 1)) / half_r[:, i].view(-1, 1, 1).clamp(min=1.0)) ** 2
             + ((cols - centre_c[:, i].view(-1, 1, 1)) / half_c[:, i].view(-1, 1, 1).clamp(min=1.0)) ** 2)
        ellipse = (1.0 - d).clamp(0.0, 1.0).sqrt() * enabled[:, i].view(-1, 1, 1)
        mask = torch.maximum(mask, ellipse.unsqueeze(1))
    return mask


def _local_patch_mask(x, s, draw_p, geometry, max_count, base_size, size_gain, one_camera):
    """Soft elliptical patches placed per route, on one camera or anywhere."""
    b, _, height, width = x.shape
    tiles = _tiles(width, geometry.num_cameras)
    count = max_count
    # How many of the max_count patches are active grows with severity.
    active = (s.view(b, 1) * max_count).ceil().clamp(min=1)
    camera = (draw_p((b,)) * geometry.num_cameras).floor().clamp(max=geometry.num_cameras - 1)
    tile_start = torch.tensor([t[0] for t in tiles], device=x.device, dtype=torch.float32)[camera.long()]
    tile_width = torch.tensor([t[1] - t[0] for t in tiles], device=x.device, dtype=torch.float32)[camera.long()]
    u_r, u_c, u_size, u_aspect = (draw_p((b, count)) for _ in range(4))
    centre_r = u_r * height
    if one_camera:
        centre_c = tile_start.view(-1, 1) + u_c * tile_width.view(-1, 1)
    else:
        centre_c = u_c * width
    size = (base_size + size_gain * s.view(b, 1)) * (0.6 + 0.4 * u_size)
    half_r = size * height
    half_c = size * (tile_width.view(-1, 1) if one_camera else width / geometry.num_cameras) * (0.7 + 0.6 * u_aspect)
    enabled = (torch.arange(count, device=x.device).view(1, -1) < active).float()
    mask = _soft_ellipses(b, height, width, count, centre_r, centre_c, half_r, half_c, enabled, x.device)
    if one_camera:
        # Keep a lens fault on its own lens even where an ellipse would spill.
        _, cols = _grid(1, width, x.device)
        inside = ((cols >= tile_start.view(-1, 1, 1)) & (cols < (tile_start + tile_width).view(-1, 1, 1)))
        mask = mask * inside.unsqueeze(1).float()
    return mask


def _cam_local_noise(x, s, generator, draw_p, geometry):
    """Low-light ISO patches: shot noise that grows with the signal, plus read noise."""
    b = x.shape[0]
    mask = _local_patch_mask(x, s, draw_p, geometry, max_count=6, base_size=0.08,
                             size_gain=0.14, one_camera=False)
    k = s.view(b, 1, 1, 1)
    # Poisson-Gaussian model: variance a*signal + b^2, both growing with severity.
    sigma = torch.sqrt((60.0 * k) * x.clamp(min=0.0) + (30.0 * k) ** 2 + 1e-6)
    noise = _draw_normal(x.shape, generator, x.device) * sigma
    return x + mask * noise


def _cam_exposure_pulse(x, s, generator, draw_p, geometry):
    """Specular flare: one to three over-exposed spots, each with a directional bloom."""
    b, _, height, width = x.shape
    rows, cols = _grid(height, width, device=x.device)
    count = 3
    active = (s.view(b, 1) * count).ceil().clamp(min=1)
    u = [draw_p((b, count)) for _ in range(4)]
    centre_r, centre_c = u[0] * height * 0.8, u[1] * width
    radius = (0.04 + 0.08 * s.view(b, 1)) * height * (0.7 + 0.6 * u[2])
    angle = u[3] * math.pi
    glow = torch.zeros(b, 1, height, width, device=x.device)
    for i in range(count):
        dr = rows - centre_r[:, i].view(-1, 1, 1)
        dc = cols - centre_c[:, i].view(-1, 1, 1)
        cos, sin = torch.cos(angle[:, i]).view(-1, 1, 1), torch.sin(angle[:, i]).view(-1, 1, 1)
        along = dr * sin + dc * cos
        across = -dr * cos + dc * sin
        rad = radius[:, i].view(-1, 1, 1)
        core = torch.exp(-(dr**2 + dc**2) / (2 * rad**2))
        bloom = 0.6 * torch.exp(-(along**2) / (2 * (4 * rad) ** 2) - (across**2) / (2 * (0.4 * rad) ** 2))
        on = (i < active[:, 0]).float().view(-1, 1, 1)
        glow = torch.maximum(glow, (on * torch.maximum(core, bloom)).unsqueeze(1))
    k = s.view(b, 1, 1, 1)
    lifted = x + 255.0 * k * glow
    # Gain-then-gamma saturation inside the glow washes detail out, not just brightens it.
    gamma = 1.0 - 0.6 * k * glow
    return 255.0 * (lifted.clamp(0.0, 255.0) / 255.0).clamp(min=1e-6) ** gamma


def _cam_local_occlusion(x, s, generator, draw_p, geometry):
    """Mud or debris on one lens: dark semi-opaque blobs with darker rims."""
    b = x.shape[0]
    mask = _local_patch_mask(x, s, draw_p, geometry, max_count=4, base_size=0.10,
                             size_gain=0.22, one_camera=True)
    k = s.view(b, 1, 1, 1)
    alpha = mask * (0.55 + 0.45 * k)
    # Edge darkening: the rim, where the mask is still low, is darker than the centre.
    rim = 4.0 * mask * (1.0 - mask)
    mud = torch.tensor([62.0, 48.0, 34.0], device=x.device).view(1, 3, 1, 1) * (1.0 - 0.5 * rim)
    return (1.0 - alpha) * x + alpha * mud


def _cam_night_lowlight(x, s, generator, draw_p, geometry):
    """Dusk or night: less light, a crushing gamma, a warm cast, less contrast."""
    b = x.shape[0]
    k = s.view(b, 1, 1, 1)
    y = (x / 255.0).clamp(0.0, 1.0) ** (1.0 + 1.0 * k)
    y = y * (1.0 - 0.8 * k)
    warm = torch.cat([1.0 + 0.08 * k, torch.ones_like(k), 1.0 - 0.2 * k], dim=1)
    y = y * warm
    mean = y.mean(dim=(1, 2, 3), keepdim=True)
    y = mean + (1.0 - 0.4 * k) * (y - mean)
    return 255.0 * y


def _line_kernel(length: int, angle: float, device) -> torch.Tensor:
    """A normalised straight-line blur kernel of odd size."""
    size = max(1, int(length)) | 1
    kernel = torch.zeros(size, size, device=device)
    centre = size // 2
    steps = max(size * 4, 8)
    for t in torch.linspace(-centre, centre, steps):
        r = int(round(centre + float(t) * math.sin(angle)))
        c = int(round(centre + float(t) * math.cos(angle)))
        if 0 <= r < size and 0 <= c < size:
            kernel[r, c] = 1.0
    return kernel / kernel.sum()


def _per_tile(x, geometry, fn):
    """Apply ``fn`` to each camera tile separately and stitch the result back."""
    out = torch.empty_like(x)
    for start, end in _tiles(x.shape[-1], geometry.num_cameras):
        out[..., start:end] = fn(x[..., start:end])
    return out


def _cam_motion_blur(x, s, generator, draw_p, geometry):
    """Ego-motion or vibration: a directional blur, direction fixed per route."""
    b = x.shape[0]
    angle = draw_p((b,)) * math.pi
    out = torch.empty_like(x)
    for i in range(b):
        length = 1 + round(24 * float(s[i]))
        if length <= 1:
            out[i] = x[i]
            continue
        kernel = _line_kernel(length, float(angle[i]), x.device)
        weight = kernel.expand(3, 1, *kernel.shape).contiguous()
        pad = kernel.shape[-1] // 2

        def blur(tile, weight=weight, pad=pad):
            padded = F.pad(tile.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")
            return F.conv2d(padded, weight, groups=3)[0]

        out[i] = _per_tile(x[i : i + 1], geometry, lambda t: blur(t[0]).unsqueeze(0))[0]
    return out


def _shift(tile: torch.Tensor, dr: int, dc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift a (c, h, w) tile; returns it and the mask of pixels that stayed in frame."""
    c, h, w = tile.shape
    out = torch.zeros_like(tile)
    valid = torch.zeros(1, h, w, device=tile.device)
    rs, re = max(0, dr), min(h, h + dr)
    cs, ce = max(0, dc), min(w, w + dc)
    out[:, rs:re, cs:ce] = tile[:, rs - dr : re - dr, cs - dc : ce - dc]
    valid[:, rs:re, cs:ce] = 1.0
    return out, valid


def _cam_ghosting(x, s, generator, draw_p, geometry):
    """Lens reflection: two or three shifted translucent copies of the frame.

    Where a copy is shifted in from outside the frame there is nothing to copy,
    and those pixels are darkened rather than invented.
    """
    b = x.shape[0]
    count = 3
    u = [draw_p((b, count)) for _ in range(3)]
    out = torch.empty_like(x)
    for i in range(b):
        k = float(s[i])
        n = 2 + int(u[2][i, 0] > 0.5)
        weight = 0.5 * k / n
        shifts = []
        for j in range(n):
            magnitude = 4 + 20 * k * float(u[0][i, j])
            angle = 2 * math.pi * float(u[1][i, j])
            shifts.append((round(magnitude * math.sin(angle)), round(magnitude * math.cos(angle))))

        def ghost(tile, shifts=shifts, weight=weight, n=n):
            result = (1.0 - weight * n) * tile
            for dr, dc in shifts:
                copy, valid = _shift(tile, dr, dc)
                result = result + weight * copy * valid
            return result

        out[i] = _per_tile(x[i : i + 1], geometry, lambda t: ghost(t[0]).unsqueeze(0))[0]
    return out


def _cam_color_shift(x, s, generator, draw_p, geometry):
    """White-balance or colour-calibration drift: a warm or cool cast, contrast, gamma."""
    b = x.shape[0]
    k = s.view(b, 1, 1, 1)
    u = draw_p((b, 3)) * 2.0 - 1.0
    temperature = u[:, 0].view(b, 1, 1, 1)  # >0 warm, <0 cool
    gains = torch.cat([1.0 + 0.25 * k * temperature,
                       1.0 + 0.05 * k * u[:, 1].view(b, 1, 1, 1),
                       1.0 - 0.25 * k * temperature], dim=1)
    y = (x / 255.0).clamp(0.0, 1.0) * gains
    mean = y.mean(dim=(1, 2, 3), keepdim=True)
    y = mean + (1.0 - 0.3 * k) * (y - mean)
    y = y.clamp(min=1e-6) ** (1.0 + 0.4 * k * u[:, 2].view(b, 1, 1, 1))
    return 255.0 * y


_CAMERA_IMPL = {
    "cam_signal_drop": (_cam_signal_drop, 11),
    "cam_local_noise": (_cam_local_noise, 12),
    "cam_exposure_pulse": (_cam_exposure_pulse, 13),
    "cam_local_occlusion": (_cam_local_occlusion, 14),
    "cam_night_lowlight": (_cam_night_lowlight, 15),
    "cam_motion_blur": (_cam_motion_blur, 16),
    "cam_ghosting": (_cam_ghosting, 17),
    "cam_color_shift": (_cam_color_shift, 18),
}


# --- LiDAR faults ------------------------------------------------------------


def _polar(height, width, geometry, device):
    """Range in metres and bearing in radians of every raster cell from the sensor."""
    rows, cols = _grid(height, width, device)
    ppm = geometry.bev_pixels_per_meter
    ego_row = -geometry.bev_min_y_meter * ppm
    ego_col = -geometry.bev_min_x_meter * ppm
    lateral = (rows - ego_row) / ppm
    longitudinal = (cols - ego_col) / ppm
    return torch.sqrt(lateral**2 + longitudinal**2), torch.atan2(lateral, longitudinal)


def _spurious(shape, density, generator, device, low=0.2, high=1.0):
    """Sparse false returns: Bernoulli occupancy with random normalised density."""
    hit = (_draw(shape, generator, device) < density).float()
    return hit * (low + (high - low) * _draw(shape, generator, device))


def _lid_signal_drop(x, s, generator, draw_p, geometry):
    """Jamming: true returns vanish, replaced by spurious ones that thin with range."""
    b, c, h, w = x.shape
    distance, _ = _polar(h, w, geometry, x.device)
    far = (distance / _FAR_RANGE_M).clamp(0.0, 1.0)
    k = s.view(b, 1, 1, 1)
    keep = (_draw(x.shape, generator, x.device) >= k).float()
    density = k * (0.05 + 0.45 * (1.0 - 0.6 * far))
    return torch.maximum(x * keep, _spurious(x.shape, density, generator, x.device))


def _lid_range_dropout(x, s, generator, draw_p, geometry):
    """Rain, fog or absorption: the sensor's effective range shrinks.

    Modelled as a visibility range, 70 m in clear air down to 15 m at full
    severity, with a soft edge: returns well inside it survive, returns past it
    are lost, plus a light thinning everywhere and backscatter near the sensor.
    A first version dropped returns with a gentle ramp in range, and at full
    severity still kept a third of the returns at 50 m -- not what fog does.
    """
    b, c, h, w = x.shape
    distance, _ = _polar(h, w, geometry, x.device)
    k = s.view(b, 1, 1, 1)
    visibility = _FAR_RANGE_M - 55.0 * k
    keep_probability = torch.sigmoid((visibility - distance) / 4.0) * (1.0 - 0.15 * k)
    keep = (_draw(x.shape, generator, x.device) < keep_probability).float()
    near = (distance < 8.0).float()
    backscatter = _spurious(x.shape, 0.03 * k * near, generator, x.device, 0.2, 0.6)
    return torch.maximum(x * keep, backscatter)


def _lid_frustum_occlusion(x, s, generator, draw_p, geometry):
    """Mud on the dome: an angular wedge fully blind, bearing fixed per route.

    The wedge points into the forward half, within 75 degrees of straight
    ahead. Mud can land anywhere on a dome, but a blind spot behind the car
    barely touches forward driving, so a uniform bearing would make this
    fault's cost vary from route to route for no reason the test cares about.
    """
    b, c, h, w = x.shape
    distance, bearing = _polar(h, w, geometry, x.device)
    centre = (draw_p((b,)) * 2.0 - 1.0) * math.radians(75.0)
    half = torch.deg2rad(8.0 + 40.0 * s)
    diff = torch.remainder(bearing.unsqueeze(0) - centre.view(-1, 1, 1) + math.pi, 2 * math.pi) - math.pi
    wedge = (diff.abs() < half.view(-1, 1, 1)).float().unsqueeze(1)
    near = (distance < 4.0).float()
    false_returns = _spurious(x.shape, 0.02 * s.view(b, 1, 1, 1) * near, generator, x.device, 0.2, 0.6)
    return x * (1.0 - wedge) + wedge * false_returns


def _lid_local_speckle(x, s, generator, draw_p, geometry):
    """Dust or insects: a few oriented clusters of false returns in empty cells."""
    b, c, h, w = x.shape
    rows, cols = _grid(h, w, x.device)
    ppm = geometry.bev_pixels_per_meter
    count = 5
    active = (2 + (s.view(b, 1) * 3).floor())
    u = [draw_p((b, count)) for _ in range(5)]
    blobs = torch.zeros(b, 1, h, w, device=x.device)
    for i in range(count):
        cr = (0.2 + 0.6 * u[0][:, i]).view(-1, 1, 1) * h
        cc = (0.2 + 0.6 * u[1][:, i]).view(-1, 1, 1) * w
        major = ((1.5 + 2.5 * u[2][:, i]) * ppm).view(-1, 1, 1)
        minor = ((0.5 + 1.0 * u[3][:, i]) * ppm).view(-1, 1, 1)
        theta = (u[4][:, i] * math.pi).view(-1, 1, 1)
        dr, dc = rows - cr, cols - cc
        along = dr * torch.sin(theta) + dc * torch.cos(theta)
        across = -dr * torch.cos(theta) + dc * torch.sin(theta)
        blob = torch.exp(-(along**2) / (2 * major**2) - (across**2) / (2 * minor**2))
        on = (i < active[:, 0]).float().view(-1, 1, 1)
        blobs = torch.maximum(blobs, (on * blob).unsqueeze(1))
    density = 0.3 * s.view(b, 1, 1, 1) * blobs
    empty = (x <= 0).float()
    return x + empty * _spurious(x.shape, density, generator, x.device)


def _lid_feature_noise(x, s, generator, draw_p, geometry):
    """Measurement noise on the cells that have returns."""
    b = x.shape[0]
    k = s.view(b, 1, 1, 1)
    occupied = (x > 0).float()
    return x + occupied * 0.35 * k * _draw_normal(x.shape, generator, x.device)


_LIDAR_IMPL = {
    "lid_signal_drop": (_lid_signal_drop, 21),
    "lid_range_dropout": (_lid_range_dropout, 22),
    "lid_frustum_occlusion": (_lid_frustum_occlusion, 23),
    "lid_local_speckle": (_lid_local_speckle, 24),
    "lid_feature_noise": (_lid_feature_noise, 25),
}


# --- entry point -------------------------------------------------------------


def apply_fault(
    batch: dict,
    fault: str,
    severity: torch.Tensor,
    generator: torch.Generator | None = None,
    persistent_seed: int | None = None,
    geometry: FaultGeometry | None = None,
) -> dict:
    """Apply one catalogue fault to a collated batch, per-sample severity.

    Args:
        batch: Collated inputs; ``rgb`` for camera faults, ``rasterized_lidar``
            for LiDAR faults. Modified in place and returned.
        fault: A name from :data:`FAULT_CATALOG`.
        severity: Per-sample strength in ``[0, 1]``, shape ``(b,)``.
        generator: The advancing stream for what varies in time.
        persistent_seed: Seeds the fault's geometry so it holds still across the
            ticks of a route; None in training.
        geometry: Input layout; LEAD's defaults if None.

    Returns:
        The batch.

    Raises:
        ValueError: For a name outside the catalogue.
    """
    geometry = geometry or FaultGeometry()
    if fault in _CAMERA_IMPL:
        key, (impl, salt) = "rgb", _CAMERA_IMPL[fault]
        low, high = 0.0, 255.0
    elif fault in _LIDAR_IMPL:
        key, (impl, salt) = "rasterized_lidar", _LIDAR_IMPL[fault]
        low, high = 0.0, 1.0
    else:
        raise ValueError(f"unknown fault '{fault}'; the catalogue is {list(FAULT_CATALOG)}.")
    if key not in batch:
        return batch
    original = batch[key]
    x = original.to(torch.float32)
    s = severity.to(x.device, torch.float32).clamp(0.0, 1.0)
    draw_p = _persistent(persistent_seed, generator, x.device, salt)
    damaged = impl(x, s, generator, draw_p, geometry).clamp(low, high)
    # A zero-severity sample is returned exactly, not merely approximately.
    untouched = (s <= 0.0).view(-1, *([1] * (x.dim() - 1)))
    damaged = torch.where(untouched, x, damaged)
    if original.dtype == torch.uint8:
        damaged = damaged.round()
    batch[key] = damaged.to(original.dtype)
    return batch
