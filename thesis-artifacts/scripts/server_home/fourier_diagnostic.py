# -*- coding: utf-8 -*-
"""Is the policy already blind to the camera's amplitude spectrum? Stage 0 of the
Fourier idea, open loop, no training.

The idea: augment training by perturbing each camera's Fourier amplitude while
keeping its phase, on the premise that phase carries the scene's geometry and
amplitude its appearance, so the waypoint label stays valid. That only has
something to fix if the trained policy's plan moves when the amplitude changes.
This measures it, on the same frames, for two checkpoints, against the moves
that bound it:

    clean           the frame as is (checks the forward pass is deterministic)
    amp_mix_0.5     luminance amplitude halfway to another frame's, phase kept
    amp_swap        luminance amplitude replaced by another frame's, phase kept
    phase_swap      luminance phase replaced by another frame's, amplitude kept
    image_swap      the whole camera image replaced by another frame's
    camera_0.5/1.0  the thesis's own camera degradation (degrade_batch)
    camera_black    no camera signal at all

Every spectral edit is per camera (the three cameras are side by side in one
image, and one FFT across them would mix them at the seams), in float32, on
luminance only: dY is added to R, G and B alike, which leaves Cb and Cr -- and
so the colour of a traffic light -- exactly as they were, up to clipping at 0
and 255. The donor frame is the next frame in the same shuffled batch.

For each condition it reports how much the input changed (mean |d pixel|, 0-255)
and how far the plan moved (mean Euclidean displacement over the future
waypoints, metres), and that move as a share of image_swap's: the image_swap
move is what changing the camera's whole content does to this model, so it is
the ceiling any appearance-only change should be read against.

Reading it:
  * amp_swap well below image_swap, near clean  -> the model already ignores the
    amplitude; the augmentation has nothing to fix.
  * amp_swap a large share of image_swap        -> the plan depends on appearance
    it should not; the augmentation has room to help.
  * phase_swap near image_swap, amp_swap not    -> the model reads geometry from
    the phase, as the premise assumes.

Frames are drawn from the seed-2 model's own training set (the 585 logs) -- the
only logs on this machine -- so this is in-sample. The quantity is a
sensitivity, not an error, which in-sample frames do not inflate the way they
inflate an accuracy; it is still a limitation and is printed as one.

Usage: python fourier_diagnostic.py [--device cuda] [--batches 40] [--batch-size 8]
                                     [--workers 8] [--out results/fourier_diagnostic.csv]
                                     NAME=CHECKPOINT_DIR [...]
The first model's dataset supplies the frames; every model sees the same ones.
"""
import argparse
import csv
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/common"))
from analyze_gate import load_model, to_device  # noqa: E402
from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

# The frames are held for the whole run so every model sees the same ones; with
# the default sharing strategy each held tensor keeps a file descriptor open, and
# forty batches run out of them ("received 0 items of ancdata").
torch.multiprocessing.set_sharing_strategy("file_system")

NUM_CAMERAS = 3
LUMA = torch.tensor([0.299, 0.587, 0.114])


def tiles(width: int):
    tile = width // NUM_CAMERAS
    return [(k * tile, (k + 1) * tile if k < NUM_CAMERAS - 1 else width) for k in range(NUM_CAMERAS)]


def spectral_edit(rgb: torch.Tensor, donor: torch.Tensor, mode: str, lam: float = 1.0) -> torch.Tensor:
    """Edit each camera's luminance spectrum; chroma untouched.

    Args:
        rgb: (B, 3, H, W), 0-255.
        donor: Same shape, the frames whose spectrum is borrowed.
        mode: "amplitude" (mix amplitude by lam, keep phase) or "phase"
            (take the donor's phase, keep amplitude).
        lam: Amplitude mixing weight; 1.0 is a full swap.

    Returns:
        The edited frames, same dtype and range as ``rgb``.
    """
    x = rgb.float()
    d = donor.float()
    w = LUMA.to(x.device).view(1, 3, 1, 1)
    y, yd = (x * w).sum(1), (d * w).sum(1)
    out = y.clone()
    for start, end in tiles(x.shape[-1]):
        f = torch.fft.fft2(y[..., start:end])
        fd = torch.fft.fft2(yd[..., start:end])
        if mode == "amplitude":
            amp = (1.0 - lam) * f.abs() + lam * fd.abs()
            new = torch.polar(amp, f.angle())
        else:
            new = torch.polar(f.abs(), fd.angle())
        out[..., start:end] = torch.fft.ifft2(new).real
    edited = (x + (out - y).unsqueeze(1)).clamp(0.0, 255.0)
    return edited.to(rgb.dtype) if rgb.dtype.is_floating_point else edited.round().to(rgb.dtype)


def plan(model, batch, device):
    batch = dict(batch)
    batch["current_gradient_step"] = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(batch).future_waypoints.float()


def conditions(batch, index, device):
    """Yield (name, edited batch) for every condition, clean first."""
    rgb = batch["rgb"]
    donor = torch.roll(rgb, shifts=1, dims=0)
    yield "clean", batch
    for name, new in (("amp_mix_0.5", spectral_edit(rgb, donor, "amplitude", 0.5)),
                      ("amp_swap", spectral_edit(rgb, donor, "amplitude", 1.0)),
                      ("phase_swap", spectral_edit(rgb, donor, "phase")),
                      ("image_swap", donor.clone()),
                      ("camera_black", torch.zeros_like(rgb))):
        yield name, {**batch, "rgb": new}
    for severity in (0.5, 1.0):
        damaged = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        generator = torch.Generator(device=device).manual_seed(index)
        yield f"camera_{severity:g}", degrade_batch(damaged, "camera", severity, generator=generator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="results/fourier_diagnostic.csv")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    device = torch.device(args.device)

    frames = None
    rows = []
    for entry in args.models:
        name, _, path = entry.partition("=")
        _, model = load_model(pathlib.Path(path), device)
        model.eval()
        if frames is None:
            dataset = model.build_dataset()
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                collate_fn=getattr(dataset, "collate_fn", None), num_workers=args.workers,
                                generator=torch.Generator().manual_seed(0))
            frames = []
            for index, batch in enumerate(loader):
                if index >= args.batches:
                    break
                frames.append(batch)
        moved, changed = {}, {}
        try:
            for index, cpu_batch in enumerate(frames):
                batch = to_device(cpu_batch, device)
                reference = plan(model, batch, device)
                for cond, edited in conditions(batch, index, device):
                    out = plan(model, edited, device)
                    moved.setdefault(cond, []).append((out - reference).norm(dim=-1).mean(dim=-1).cpu())
                    changed.setdefault(cond, []).append(
                        float((edited["rgb"].float() - batch["rgb"].float()).abs().mean()))
        except Exception as error:  # noqa: BLE001 -- report and go on to the next model
            print(f"{name}: failed on these frames: {type(error).__name__}: {error}")
            continue
        ceiling = float(torch.cat(moved["image_swap"]).mean())
        print(f"\n=== {name}  ({sum(len(m) for m in moved['clean'])} frames, in-sample)")
        print(f"{'condition':14s} {'|d pixel|':>10s} {'moved m':>9s} {'median':>8s} {'of image_swap':>14s}")
        for cond in moved:
            shift = torch.cat(moved[cond])
            share = float(shift.mean()) / ceiling if ceiling > 0 else float("nan")
            change = sum(changed[cond]) / len(changed[cond])
            print(f"{cond:14s} {change:10.2f} {float(shift.mean()):9.3f} {float(shift.median()):8.3f} {share:14.2f}")
            rows.append({"model": name, "condition": cond, "frames": shift.numel(),
                         "input_change": round(change, 3), "moved_mean_m": round(float(shift.mean()), 4),
                         "moved_median_m": round(float(shift.median()), 4),
                         "share_of_image_swap": round(share, 4)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
