"""What a finer fusion token grid would cost, before anything is trained.

The deformable operator loses at this architecture's 552 tokens and wins at
2208; the operator benchmark says so and results/cost.csv confirms the loss end
to end. So the question worth asking is not how to make fusion cheaper -- the
fusion blocks are a small part of the dense model's forward pass, too small for
a cheaper operator to return much -- but whether the finer token grid the sparse
operator makes affordable is worth having at all.

Answering that by training costs forty hours a rung. Answering this much of it
costs minutes, and it can refuse the rung before the GPU is booked.

WHAT IS MEASURED, AND WHAT IS ASSUMED

Measured rather than argued:

1. The real feature-map shape and channel width entering each fusion block. The
   backbone fuses four times, once per encoder stage, and each stage hands
   ``fuse_features`` a different resolution and a different channel count. The
   existing operator benchmark sweeps one channel width for all of them, which
   is fine for comparing operators against each other and wrong for predicting
   this model's cost.

2. Both operators timed at each stage's own token count and channel width, at
   today's grid and at the finer one. Today's timing is not redundant: it is
   what separates a fusion block's attention from everything else in it.

3. The whole forward pass, so per-block numbers become a share. An operator
   three times cheaper inside a part that is 2% of the model is worth nothing,
   and only the share tells you which case you are in.

Assumed, and worth stating because the prediction rests on it: everything
outside the fusion blocks costs what it costs today. That holds because the
anchor grid is a pooling decision applied to features the encoders have already
produced, so refining it changes what enters fusion, not what the encoders
compute.

THE CAP, WHICH IS THE POINT

A finer anchor grid does not buy the same thing at every stage. Stage 0 hands
fusion a stride-4 map and the anchor grid throws away 8x8 blocks of it. Stage 3
hands over a stride-32 map the anchor grid already matches exactly, so asking it
for a finer grid upsamples: more tokens, more attention, no new information.

Each stage's grid is therefore capped at that stage's own resolution, and the
table reports where the cap bites. A prediction ignoring the cap would overstate
both the cost and the benefit, and an implementation ignoring it would spend the
cost to get none of the benefit.
"""

import argparse
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402
from forward_profile import profile  # noqa: E402
from interleaved_timing import (  # noqa: E402
    build_probe,
    check_replica,
    time_interleaved,
)

# Below this share, an operator three times cheaper inside fusion still returns
# almost nothing end to end, and the sparse rung is not worth its forty hours.
# The dense finer-grid rung can still be, because it answers a different
# question: whether the geometry is worth anything.
_SHARE_WORTH_A_RUNG = 0.15


def stage_geometry(model, lead_config, loader, device):
    """The feature map each fusion block actually receives, per stage.

    The two pooling modules are shared across the four stages and called once
    each, so a pre-hook fires four times per forward pass and the calls arrive
    in stage order.

    Args:
        model: The loaded policy.
        lead_config: Its config, for the autocast dtype. The weights are fp32
            and the batch arrives in the training dtype, so a forward pass
            outside autocast fails on the first convolution.
        loader: Loader over the probe frames.
        device: Device to run on.

    Returns:
        A list of ``(image_shape, lidar_shape, channels)`` per stage, the shapes
        being ``(rows, cols)`` of the map before pooling.

    Raises:
        SystemExit: If the backbone does not pool to an anchor grid, or the two
            pools are called different numbers of times. Either means the
            architecture moved and everything below would be quietly wrong.
    """
    backbone = getattr(model, "backbone", None)
    image_pool = getattr(backbone, "avgpool_img", None)
    lidar_pool = getattr(backbone, "avgpool_lidar", None)
    if image_pool is None or lidar_pool is None:
        raise SystemExit(
            "this backbone has no avgpool_img/avgpool_lidar, so it does not pool "
            "to an anchor grid and this prediction does not apply to it.",
        )

    image_shapes: list[tuple[int, int, int]] = []
    lidar_shapes: list[tuple[int, int]] = []

    def note_image(_module, inputs):
        tensor = inputs[0]
        image_shapes.append((tensor.shape[2], tensor.shape[3], tensor.shape[1]))

    def note_lidar(_module, inputs):
        tensor = inputs[0]
        lidar_shapes.append((tensor.shape[2], tensor.shape[3]))

    handles = [
        image_pool.register_forward_pre_hook(note_image),
        lidar_pool.register_forward_pre_hook(note_lidar),
    ]
    batch = to_device(next(iter(loader)), device)
    model.eval()
    optimization = lead_config.training.optimization
    with torch.inference_mode(), autocast(
        device_type=device.type,
        dtype=optimization.torch_dtype,
        enabled=optimization.use_mixed_precision_training,
    ):
        model(batch)
    for handle in handles:
        handle.remove()

    if not image_shapes or len(image_shapes) != len(lidar_shapes):
        raise SystemExit(
            f"expected both pools to be called the same number of times; got "
            f"{len(image_shapes)} image and {len(lidar_shapes)} lidar calls.",
        )
    return [
        ((rows, cols), lidar_shapes[index], channels)
        for index, (rows, cols, channels) in enumerate(image_shapes)
    ]


def capped_grid(requested, native):
    """A stage's anchor grid, never finer than the map it pools.

    Args:
        requested: ``(rows, cols)`` asked for.
        native: ``(rows, cols)`` of the stage's own feature map.

    Returns:
        The grid to use, and whether the cap changed the request.
    """
    grid = (min(requested[0], native[0]), min(requested[1], native[1]))
    return grid, grid != tuple(requested)


def time_all_stages(today_plan, finer_plan, config, arguments, device):
    """Time every stage's operators in one interleaved pass.

    Timing each configuration to completion in turn is what produced the two
    unusable runs before this: on a shared card a contiguous window inherits
    whatever else ran during it, and the worst case was one configuration
    measured at 1.13 ms and 9.44 ms inside a single process. Every
    configuration in this script is therefore built up front and handed to
    ``time_interleaved``, which visits them in rotating rounds and takes the
    median over rounds rather than the mean within one window.

    A replica of one configuration goes in alongside the real ones. It is a
    separately constructed module of identical shape, so the two medians are
    measuring the same computation and any disagreement is the method failing
    rather than a result.

    Args:
        today_plan: Per-stage plan at the shipped grid.
        finer_plan: Per-stage plan at the requested grid.
        config: The transfuser config, for the head count.
        arguments: Parsed arguments.
        device: Device to run on.

    Returns:
        ``(results, replica_gap)`` where results maps
        ``(when, stage, operator)`` to a :class:`Timing`.

    Raises:
        SystemExit: Via :func:`check_replica`, if the method is not working on
            this machine today.
    """
    probes = {}
    for when, plan in (("now", today_plan), ("finer", finer_plan)):
        for index, tokens, channels, (image_grid, bev_grid), _ in plan:
            for kind in ("dense", "deformable"):
                probes[f"{when}_{index}_{kind}"] = build_probe(
                    kind,
                    tokens,
                    channels,
                    image_grid,
                    bev_grid,
                    config,
                    arguments.batch_size,
                    arguments.num_points,
                    device,
                    arguments.compile,
                )
    # The control. Same shape as stage 0 at the shipped grid, built separately.
    index0, tokens0, channels0, (image0, bev0), _ = today_plan[0]
    probes["replica"] = build_probe(
        "dense", tokens0, channels0, image0, bev0, config,
        arguments.batch_size, arguments.num_points, device, arguments.compile,
    )

    results = time_interleaved(
        probes, arguments.rounds, arguments.iterations, arguments.warmup,
    )
    gap = check_replica(results, f"now_{index0}_dense", "replica")
    return results, gap


def check_monotonic(label, series, tolerance=1.25):
    """Refuse timings that cannot be right, instead of concluding from them.

    Within one token count, an attention operator cannot get cheaper as its
    channel width grows. When it does, the measurement is noise and every
    number downstream of it -- the totals, the shares, the verdict -- is noise
    wearing a decimal point. This is the check the first version of this script
    did not have: it printed a confident verdict computed from timings that were
    not monotonic in channels and disagreed with an independent benchmark of the
    same configuration by a factor of four.

    Args:
        label: What is being checked, for the message.
        series: ``[(channels, milliseconds)]`` at one fixed token count.
        tolerance: How much of an inversion to forgive, as a ratio. Timing noise
            of a few percent is expected; a wider channel measuring 25% cheaper
            than a narrower one is not noise.

    Raises:
        SystemExit: If the series inverts by more than the tolerance.
    """
    ordered = sorted(series)
    if len(ordered) < 3:
        return
    for (narrow_c, narrow_ms), (wide_c, wide_ms) in zip(
        ordered, ordered[1:], strict=False,
    ):
        if narrow_ms > wide_ms * tolerance:
            raise SystemExit(
                f"{label}: {narrow_c} channels measured {narrow_ms:.2f} ms and "
                f"{wide_c} channels measured {wide_ms:.2f} ms. A wider operator "
                f"cannot be cheaper, so these timings are noise and nothing "
                f"computed from them would mean anything. Re-run on an idle "
                f"device; if it persists, the timing method is wrong. "
                f"Full series: {ordered}",
            )


def plan_grids(stages, wanted_image, wanted_bev):
    """Per-stage grids and token counts under the cap.

    Args:
        stages: Output of :func:`stage_geometry`.
        wanted_image: ``(rows, cols)`` asked for on the image side.
        wanted_bev: ``(rows, cols)`` asked for on the BEV side.

    Returns:
        A list of ``(index, tokens, channels, (image_grid, bev_grid), capped)``.
    """
    plan = []
    for index, (image_map, lidar_map, channels) in enumerate(stages):
        image_grid, image_capped = capped_grid(wanted_image, image_map)
        bev_grid, bev_capped = capped_grid(wanted_bev, lidar_map)
        tokens = image_grid[0] * image_grid[1] + bev_grid[0] * bev_grid[1]
        plan.append(
            (index, tokens, channels, (image_grid, bev_grid), image_capped or bev_capped),
        )
    return plan


def main() -> int:
    """Predict the cost of a finer grid and say whether it earns a rung.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="DIR")
    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=16,
        help="the stride the anchor grid would be taken at; today's is 32",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile-warmup", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=40)
    arguments = parser.parse_args()

    if arguments.device != "cuda":
        print("CUDA events are the only timer here; run this on the GPU.")
        return 1

    device = torch.device(arguments.device)
    lead_config, model = load_model(pathlib.Path(arguments.model), device)
    config = lead_config.policy.transfuser
    dataset = model.build_dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=getattr(dataset, "collate_fn", None),
        num_workers=arguments.workers,
    )

    stages = stage_geometry(model, lead_config, loader, device)
    per_module, whole_ms = profile(
        model,
        lead_config,
        loader,
        arguments.profile_warmup,
        arguments.profile_repeats,
        device,
    )
    fusion_now_ms = sum(v for k, v in per_module.items() if k.startswith("fusion."))
    if fusion_now_ms <= 0:
        raise SystemExit(
            "the profile attributed no time to fusion blocks; the hooks did not "
            "fire and the prediction would divide by nothing.",
        )

    today_image = (config.img_vert_anchors, config.img_horz_anchors)
    today_bev = (config.lidar_bev_grid_rows, config.lidar_bev_grid_cols)
    stride = arguments.anchor_stride
    wanted_image = (
        config.final_image_height // stride,
        config.final_image_width // stride,
    )
    wanted_bev = (
        config.lidar_height_pixel // stride,
        config.lidar_width_pixel // stride,
    )

    today_plan = plan_grids(stages, today_image, today_bev)
    finer_plan = plan_grids(stages, wanted_image, wanted_bev)

    print(
        f"\nmodel={arguments.model}  batch={arguments.batch_size}  "
        f"{'compiled' if arguments.compile else 'eager'}",
    )
    tokens_now = today_image[0] * today_image[1] + today_bev[0] * today_bev[1]
    print(
        f"today: image {today_image} + bev {today_bev} = {tokens_now} tokens, "
        f"the same at all {len(stages)} fusion blocks",
    )
    print(f"asked: stride {stride} -> image {wanted_image} + bev {wanted_bev}\n")

    header = (
        f"  {'stage':<6}{'image map':>12}{'bev map':>10}{'chans':>7}"
        f"{'image grid':>12}{'bev grid':>10}{'tokens':>8}  cap"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (index, tokens, channels, (image_grid, bev_grid), capped), stage in zip(
        finer_plan, stages, strict=True,
    ):
        print(
            f"  {index:<6}{str(stage[0]):>12}{str(stage[1]):>10}{channels:>7}"
            f"{str(image_grid):>12}{str(bev_grid):>10}{tokens:>8}  "
            f"{'capped' if capped else ''}",
        )

    print("\n  operator cost per block, at each stage's own tokens and width")
    header = (
        f"  {'stage':<6}{'tokens now':>12}{'dense':>9}{'deform':>9}"
        f"{'tokens finer':>14}{'dense':>9}{'deform':>9}{'speedup':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    timings, replica_gap = time_all_stages(
        today_plan, finer_plan, config, arguments, device,
    )

    now_dense = now_deformable = finer_dense = finer_deformable = 0.0
    worst_inflation = 1.0
    # Collected per token count, because monotonicity in channels only means
    # anything between blocks doing the same amount of work.
    by_tokens: dict[int, dict[str, list[tuple[int, float]]]] = {}

    def note(tokens, operator, channels, milliseconds):
        by_tokens.setdefault(tokens, {}).setdefault(operator, []).append(
            (channels, milliseconds),
        )

    for now, finer in zip(today_plan, finer_plan, strict=True):
        index, tokens_a, channels, _, _ = now
        _, tokens_b, _, _, _ = finer
        dense_a = timings[f"now_{index}_dense"]
        deform_a = timings[f"now_{index}_deformable"]
        dense_b = timings[f"finer_{index}_dense"]
        deform_b = timings[f"finer_{index}_deformable"]
        worst_inflation = max(
            worst_inflation,
            dense_a.inflation, deform_a.inflation,
            dense_b.inflation, deform_b.inflation,
        )
        now_dense += dense_a.best
        now_deformable += deform_a.best
        finer_dense += dense_b.best
        finer_deformable += deform_b.best
        note(tokens_a, "dense", channels, dense_a.best)
        note(tokens_a, "deformable", channels, deform_a.best)
        note(tokens_b, "dense", channels, dense_b.best)
        note(tokens_b, "deformable", channels, deform_b.best)
        print(
            f"  {index:<6}{tokens_a:>12}{dense_a.best:>9.2f}"
            f"{deform_a.best:>9.2f}{tokens_b:>14}{dense_b.best:>9.2f}"
            f"{deform_b.best:>9.2f}{dense_b.best / deform_b.best:>9.2f}x",
        )

    print(
        f"\n  cheapest of {arguments.rounds} interleaved rounds, which is the "
        f"round that came nearest to\n  having the card alone. Two separately "
        f"built copies of one configuration agree to\n  {100 * replica_gap:.1f}%; "
        f"the most contended configuration had a median "
        f"{worst_inflation:.1f}x its cheapest.",
    )
    # Refuse before concluding. Everything below is sums of these numbers.
    for tokens, operators in sorted(by_tokens.items()):
        for operator, series in sorted(operators.items()):
            check_monotonic(f"{operator} at {tokens} tokens", series)

    # A fusion block is not only its attention: there is a feed-forward, two
    # norms and the channel projections around it. The profile timed the whole
    # block; the benchmark times the attention alone. The difference is what the
    # rest of the block costs today, and it grows linearly with tokens where
    # attention grows quadratically -- so it is scaled by the token ratio rather
    # than held fixed, which would understate the finer grid's cost.
    other_now_ms = max(fusion_now_ms - now_dense, 0.0)
    token_ratio = sum(p[1] for p in finer_plan) / max(sum(p[1] for p in today_plan), 1)
    other_finer_ms = other_now_ms * token_ratio
    rest_ms = whole_ms - fusion_now_ms

    print(f"\n  whole forward pass now:    {whole_ms:8.2f} ms")
    print(
        f"  fusion blocks now:         {fusion_now_ms:8.2f} ms  "
        f"({100 * fusion_now_ms / whole_ms:.1f}%)",
    )
    print(f"    of which attention:      {now_dense:8.2f} ms")
    print(f"    of which everything else:{other_now_ms:8.2f} ms")
    print(f"  outside fusion:            {rest_ms:8.2f} ms  (assumed unchanged)")

    header = (
        f"\n  {'configuration':<26}{'fusion ms':>11}{'total ms':>11}"
        f"{'share':>9}{'vs today':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 3))
    rows = [
        ("today, dense", fusion_now_ms),
        (f"stride {stride}, dense", finer_dense + other_finer_ms),
        (f"stride {stride}, deformable", finer_deformable + other_finer_ms),
    ]
    for label, fusion_ms in rows:
        total = rest_ms + fusion_ms
        print(
            f"  {label:<26}{fusion_ms:>11.2f}{total:>11.2f}"
            f"{100 * fusion_ms / total:>8.1f}%{total / whole_ms:>9.2f}x",
        )

    finer_dense_total = rest_ms + finer_dense + other_finer_ms
    dense_share = (finer_dense + other_finer_ms) / finer_dense_total
    saved_ms = finer_dense - finer_deformable

    print("\n  --- verdict ---")
    if dense_share < _SHARE_WORTH_A_RUNG:
        print(
            f"  Dense fusion at stride {stride} would be "
            f"{100 * dense_share:.1f}% of the forward pass, under the "
            f"{100 * _SHARE_WORTH_A_RUNG:.0f}% bar.",
        )
        print(
            "  A cheaper operator cannot return much from there. Train the dense "
            "finer-grid rung if",
        )
        print(
            "  the geometry itself is worth testing; do not train the sparse one "
            "for cost.",
        )
    else:
        print(
            f"  Dense fusion at stride {stride} would be "
            f"{100 * dense_share:.1f}% of the forward pass, and the sparse "
            f"operator returns",
        )
        print(
            f"  {saved_ms:.2f} ms of it -- "
            f"{100 * saved_ms / finer_dense_total:.1f}% end to end.",
        )
        print(
            "  Both rungs are worth training, dense first: it answers whether the "
            "grid is worth",
        )
        print("  anything without the operator confounded into it.")

    print(
        "\n  Forward-pass numbers, one batch. Training also carries the backward "
        "pass, so measure",
    )
    print("  a step before booking the hours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
