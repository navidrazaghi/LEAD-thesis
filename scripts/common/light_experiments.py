"""The three experiments the defence review asked for that need no simulator.

Each answers a question the thesis currently has to leave open.

``cost``
    What the fusion operator actually costs in wall clock and memory. The
    thesis argues the deformable operator is linear where dense attention is
    quadratic, then says -- correctly -- that an asymptotic property is not a
    speedup. This measures the difference at the architecture's real size.

``intervention``
    Whether the model *relies* less on a degraded modality, rather than merely
    weighting it less. Attention weight and reliance are not the same thing: a
    weight can fall while the output does not move, if the values it multiplies
    were small anyway. So one modality's input is replaced with noise and the
    resulting movement of the predicted waypoints is measured. A model that has
    genuinely shifted its reliance should move less when the modality it has
    already discounted is destroyed.

``ood``
    Whether robustness survives corruption the model never trained on. The
    training curriculum blurs, dims and adds noise to the camera and drops
    LiDAR returns independently; every evaluation so far has used those same
    operators, so it measures generalisation inside a familiar family. These
    five are deliberately outside it.

Nothing here writes into the thesis. Each experiment prints a table and writes
a CSV; the numbers are transcribed by hand, so a mistake in this script cannot
silently become a claim.
"""

import argparse
import csv
import pathlib
import sys
import time

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# The probe already knows how to rebuild a trained policy and move a batch, and
# duplicating either would be a second place for them to drift.
from analyze_gate import load_model, to_device  # noqa: E402

from lead.policy.transfuser.utils.sensor_degradation import (  # noqa: E402
    degrade_batch,
)

# Fixed everywhere, so two models always meet identical damage and identical
# noise on identical frames.
_SEED = 20260820


def loader_for(model, batch_size: int, workers: int):
    """A loader over the probe frames, in a fixed order.

    The model builds its own dataset, so the frames here are the same ones the
    mechanism probe reads and no second construction can disagree with it.

    Args:
        model: The loaded policy.
        batch_size: Samples per batch.
        workers: Loader workers.

    Returns:
        The dataloader.
    """
    dataset = model.build_dataset()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        num_workers=workers,
    )


def forward(model, lead_config, batch):
    """One forward pass under the model's own precision settings.

    Args:
        model: The policy.
        lead_config: Its config.
        batch: An already-on-device batch.

    Returns:
        The prediction.
    """
    optimization = lead_config.training.optimization
    with autocast(
        device_type="cuda",
        dtype=optimization.torch_dtype,
        enabled=optimization.use_mixed_precision_training,
    ):
        return model(batch)


# --------------------------------------------------------------------- cost


def run_cost(models, args, device) -> list:
    """Time a forward pass, and record peak memory and parameter count.

    The synchronisation is the whole measurement: CUDA calls return before the
    work is done, so timing without it measures how fast Python can queue work.

    Args:
        models: ``[(name, checkpoint_path), ...]``.
        args: Parsed arguments.
        device: Where to run.

    Returns:
        One row per model.
    """
    rows = []
    for name, path in models:
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        loader = loader_for(model, args.batch_size, args.workers)
        batch = to_device(next(iter(loader)), device)

        parameters = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            for _ in range(args.warmup):
                forward(model, lead_config, batch)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                forward(model, lead_config, batch)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000.0)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        mean = sum(times) / len(times)
        spread = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
        rows.append({
            "model": name,
            "forward_ms_mean": round(mean, 2),
            "forward_ms_sd": round(spread, 2),
            "peak_memory_gb": round(peak, 2),
            "parameters_m": round(parameters / 1e6, 2),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
        })
        print(f"  {name:8} {mean:7.2f} ± {spread:5.2f} ms   "
              f"{peak:5.2f} GB   {parameters / 1e6:6.2f} M")
        del model
        torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------- intervention


def replace_with_noise(batch: dict, modality: str, generator) -> dict:
    """Replace one modality's input with noise of the same shape and scale.

    This is an intervention, not a degradation: the point is not to simulate a
    plausible failure but to remove the modality's information entirely, so the
    movement of the output measures how much the model was using it.

    Args:
        batch: The batch, copied shallowly before modification.
        modality: ``"camera"`` or ``"lidar"``.
        generator: Seeded generator, so every model meets the same noise.

    Returns:
        A batch with that modality replaced.
    """
    out = dict(batch)
    key = "rgb" if modality == "camera" else "rasterized_lidar"
    if key not in out:
        return out
    reference = out[key]
    noise = torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=torch.float32,
    )
    # Match the modality's own scale, so the replacement is uninformative
    # rather than merely out of range.
    scaled = noise * reference.float().std() + reference.float().mean()
    out[key] = scaled.clamp(reference.float().min(), reference.float().max()).to(
        reference.dtype)
    return out


def run_intervention(models, args, device) -> list:
    """Measure how far the predicted waypoints move when a modality is removed.

    Args:
        models: ``[(name, checkpoint_path), ...]``.
        args: Parsed arguments.
        device: Where to run.

    Returns:
        One row per model, condition and replaced modality.
    """
    rows = []
    conditions = [("none", 0.0), ("camera", 1.0), ("lidar", 1.0)]
    for name, path in models:
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        loader = loader_for(model, args.batch_size, args.workers)

        for modality, severity in conditions:
            for replaced in ("camera", "lidar"):
                damage = torch.Generator(device=device)
                damage.manual_seed(_SEED)
                noise = torch.Generator(device=device)
                noise.manual_seed(_SEED)
                movements = []
                failure = ""
                try:
                  with torch.no_grad():
                    for index, batch in enumerate(loader):
                        if index >= args.batches:
                            break
                        batch = to_device(batch, device)
                        batch = degrade_batch(batch, modality, severity, damage)
                        intact = forward(model, lead_config, batch)
                        ablated = forward(
                            model, lead_config,
                            replace_with_noise(batch, replaced, noise))
                        a = getattr(intact, "future_waypoints", None)
                        b = getattr(ablated, "future_waypoints", None)
                        if a is None or b is None:
                            continue
                        movements.append(
                            (a.float() - b.float()).norm(dim=-1).mean().item())
                except Exception as error:  # noqa: BLE001
                    failure = f"{type(error).__name__}: {error}"
                    print(f"    failed: {failure}")
                value = sum(movements) / len(movements) if movements else None
                rows.append({
                    "model": name,
                    "condition": f"{modality}:{severity}",
                    "replaced": replaced,
                    "waypoint_movement_m": round(value, 4) if value is not None else None,
                    "batches": len(movements),
                    "failure": failure,
                })
                print(f"  {name:8} {modality:7}:{severity:<4} "
                      f"replace {replaced:7} -> "
                      f"{'n/a' if value is None else f'{value:.4f}'} m")
        del model
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------- ood


def motion_blur(image: torch.Tensor, length: int = 15) -> torch.Tensor:
    """Directional blur, which the training curriculum never applies.

    Horizontal only: a car's own motion smears the world sideways far more than
    vertically, and a separable one-row kernel keeps this cheap.
    """
    channels = image.shape[-3]
    # expand() shares storage, and conv2d wants a real weight tensor.
    kernel = torch.full(
        (channels, 1, 1, length), 1.0 / length,
        device=image.device, dtype=torch.float32).contiguous()
    flat = image.float().reshape(-1, channels, image.shape[-2], image.shape[-1])
    blurred = torch.nn.functional.conv2d(
        flat, kernel, padding=(0, length // 2), groups=channels)
    return blurred.reshape(image.shape).clamp(0.0, 255.0).to(image.dtype)


def posterise(image: torch.Tensor, levels: int = 6) -> torch.Tensor:
    """Coarse intensity quantisation, standing in for heavy lossy compression."""
    step = 255.0 / levels
    return (torch.round(image.float() / step) * step).clamp(0.0, 255.0).to(image.dtype)


def occlude(image: torch.Tensor, fraction: float = 0.35) -> torch.Tensor:
    """A rectangular blind spot, as a lens obstruction would make."""
    out = image.clone()
    height, width = out.shape[-2], out.shape[-1]
    box_h, box_w = int(height * fraction), int(width * fraction)
    top, left = (height - box_h) // 2, (width - box_w) // 2
    out[..., top:top + box_h, left:left + box_w] = 0
    return out


def misalign(raster: torch.Tensor, fraction: float = 0.04) -> torch.Tensor:
    """Translate the LiDAR raster, as a calibration error would.

    The shift is a fraction of the grid rather than a fixed cell count, so it
    stays a comparable amount of world regardless of the raster's resolution.
    """
    shift = max(1, int(round(raster.shape[-1] * fraction)))
    return torch.roll(raster, shifts=(shift, shift), dims=(-2, -1))


def band_dropout(raster: torch.Tensor, fraction: float = 0.3) -> torch.Tensor:
    """Drop a contiguous band of the raster, not independent cells.

    The training curriculum drops each occupied cell on its own coin, so what
    the model has never seen is a whole connected region going missing at once.
    That is the corruption here. It is a band of the grid rather than an
    angular sector from the ego, because the ego's position in the raster is
    not assumed anywhere else in this script and would be a guess here.
    """
    out = raster.clone()
    width = out.shape[-1]
    start = int(width * (0.5 - fraction / 2))
    out[..., :, start:start + int(width * fraction)] = 0
    return out


_OOD = {
    "motion_blur": ("rgb", motion_blur),
    "posterise": ("rgb", posterise),
    "occlusion": ("rgb", occlude),
    "misalignment": ("rasterized_lidar", misalign),
    "band_dropout": ("rasterized_lidar", band_dropout),
}


def run_ood(models, args, device) -> list:
    """Waypoint error under five corruptions absent from the training family.

    Args:
        models: ``[(name, checkpoint_path), ...]``.
        args: Parsed arguments.
        device: Where to run.

    Returns:
        One row per model and corruption.
    """
    rows = []
    for name, path in models:
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        loader = loader_for(model, args.batch_size, args.workers)

        for label, (key, operator) in [("clean", (None, None))] + list(_OOD.items()):
            errors = []
            failure = ""
            try:
                with torch.no_grad():
                    for index, batch in enumerate(loader):
                        if index >= args.batches:
                            break
                        batch = to_device(batch, device)
                        if key is not None and key in batch:
                            batch = dict(batch)
                            batch[key] = operator(batch[key])
                        prediction = forward(model, lead_config, batch)
                        predicted = getattr(prediction, "future_waypoints", None)
                        if predicted is None or "future_waypoints" not in batch:
                            continue
                        target = batch["future_waypoints"].float()
                        predicted = predicted.float()
                        horizon = min(predicted.shape[1], target.shape[1])
                        errors.append(
                            (predicted[:, :horizon] - target[:, :horizon])
                            .norm(dim=-1).mean().item())
            except Exception as error:  # noqa: BLE001
                failure = f"{type(error).__name__}: {error}"
                print(f"    {label} failed: {failure}")
            value = sum(errors) / len(errors) if errors else None
            rows.append({
                "model": name,
                "corruption": label,
                "waypoint_l2_m": round(value, 4) if value is not None else None,
                "batches": len(errors),
                "failure": failure,
            })
            print(f"  {name:8} {label:14} -> "
                  f"{'n/a' if value is None else f'{value:.4f}'} m")
        del model
        torch.cuda.empty_cache()
    return rows


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("cost", "intervention", "ood"))
    parser.add_argument("--models", nargs="+", required=True,
                        help="name=checkpoint-directory")
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one experiment and write its table."""
    args = parse_args()
    device = torch.device("cuda")
    torch.manual_seed(_SEED)
    models = [tuple(entry.split("=", 1)) for entry in args.models]

    print(f"== {args.experiment} ==")
    runner = {"cost": run_cost, "intervention": run_intervention, "ood": run_ood}
    rows = runner[args.experiment](models, args, device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
