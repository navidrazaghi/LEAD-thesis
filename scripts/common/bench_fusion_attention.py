"""Benchmark the fusion attention operators against each other.

Measures the dense operator the base backbone uses against the deformable one
in ``backbone_deformable_fusion``, over a sweep of anchor-grid strides. Stride
32 is the geometry the shipped config produces; the finer strides are what the
deformable operator makes affordable.

Run it on the GPU you train on -- the operators' relative cost is dominated by
kernel behaviour, so CPU numbers do not transfer, and compiled numbers do not
follow from eager ones either.

    LEAD_RUNTIME_TYPE_CHECKING=false \
        python scripts/common/bench_fusion_attention.py --device cuda --compile

Clearing the flag is required with --compile: beartype and Dynamo cannot run
together, the same reason scripts/common/pretrain.sh clears it.
"""

from __future__ import annotations

import argparse
import time

import torch

from lead.config import load_lead_config
from lead.policy.transfuser.encoder.deformable_attention import (
    MultiScaleDeformableAttention,
)
from lead.policy.transfuser.encoder.transfuser_backbone import SelfAttention


def _synchronize(device: torch.device) -> None:
    """Wait for queued work so the timer measures the operator, not the launch."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    module: torch.nn.Module,
    x: torch.Tensor,
    iterations: int,
    warmup: int,
) -> float:
    """Median-free mean latency of a module's forward pass, in milliseconds.

    Args:
        module: The module to time, already on the right device and in eval mode.
        x: Input batch.
        iterations: Timed iterations.
        warmup: Untimed iterations run first.

    Returns:
        Mean latency per forward pass in milliseconds.
    """
    device = x.device
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            module(x)
        _synchronize(device)
        elapsed = time.perf_counter() - start
    return elapsed / iterations * 1e3


def parse_args() -> argparse.Namespace:
    """Parse the benchmark's command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--channels",
        type=int,
        default=256,
        help="Fusion width; the four stages of a resnet34 backbone use 64/128/256/512.",
    )
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[32, 24, 16, 12, 8],
        help="Anchor-grid strides to sweep. The config's own geometry is 32.",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument(
        "--compile", action="store_true", help="Wrap both in torch.compile."
    )
    return parser.parse_args()


def main() -> None:
    """Run the sweep and print a comparison table."""
    args = parse_args()
    device = torch.device(args.device)
    config = load_lead_config().policy.transfuser
    dtype_note = "compiled" if args.compile else "eager"

    image_height = config.final_image_height
    image_width = config.final_image_width
    bev_height = config.lidar_height_pixel
    bev_width = config.lidar_width_pixel

    print(
        f"device={device}  batch={args.batch_size}  channels={args.channels}  "
        f"heads={config.n_head}  points={args.num_points}  ({dtype_note})",
    )
    print(f"image {image_height}x{image_width}   bev raster {bev_height}x{bev_width}\n")
    header = (
        f"{'stride':>7} {'image grid':>12} {'bev grid':>10} {'tokens':>7} "
        f"{'dense ms':>10} {'deform ms':>11} {'speedup':>9} {'score ratio':>12}"
    )
    print(header)
    print("-" * len(header))

    for stride in args.strides:
        image_shape = (image_height // stride, image_width // stride)
        bev_shape = (bev_height // stride, bev_width // stride)
        spatial_shapes = (image_shape, bev_shape)
        tokens = image_shape[0] * image_shape[1] + bev_shape[0] * bev_shape[1]

        x = torch.randn(args.batch_size, tokens, args.channels, device=device)
        dense = SelfAttention(args.channels, config.n_head, 0.0, 0.0).to(device).eval()
        deformable = (
            MultiScaleDeformableAttention(
                n_embd=args.channels,
                n_head=config.n_head,
                attn_pdrop=0.0,
                resid_pdrop=0.0,
                spatial_shapes=spatial_shapes,
                num_points=args.num_points,
            )
            .to(device)
            .eval()
        )
        if args.compile:
            dense = torch.compile(dense)
            deformable = torch.compile(deformable)

        dense_ms = benchmark(dense, x, args.iterations, args.warmup)
        deformable_ms = benchmark(deformable, x, args.iterations, args.warmup)
        # Pairwise scores the dense operator computes, over sampled reads.
        score_ratio = tokens / (2 * args.num_points)

        print(
            f"{stride:>7} {str(image_shape):>12} {str(bev_shape):>10} {tokens:>7} "
            f"{dense_ms:>10.2f} {deformable_ms:>11.2f} "
            f"{dense_ms / deformable_ms:>8.2f}x {score_ratio:>11.1f}x",
        )


if __name__ == "__main__":
    main()
