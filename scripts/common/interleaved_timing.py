"""Timing that survives a shared GPU, and a harness that says whether it does.

THE PROBLEM THIS SOLVES

Timing an operator by running it fifty times and taking the mean assumes the
card is yours for those fifty runs. On this machine it is not: nvidia-smi shows
another user's process resident, and utilisation read 34% in a window taken
immediately after stopping our own training. There is no idle A100 to wait for.

A contiguous measurement inherits whatever happened during its window. That is
how the first pre-flight reported a 64-channel attention block as twenty-five
times more expensive than a 128-channel one, and how one configuration timed
twice inside a single process came out at 1.13 ms and 9.44 ms. Both are the same
failure: a burst of someone else's work landed inside one window and not the
other, and the mean has no defence against it.

THE METHOD

Two changes, and neither is exotic.

*Interleave.* Instead of measuring A to completion and then B, measure A once,
then B once, then A again, and so on. A burst then lands on whichever
configurations happen to be running, rather than on whichever one was unlucky
enough to be scheduled during it. The order rotates each round so no
configuration permanently occupies the position that pays for the round's
startup.

*Take the quietest round, not the mean.* Contention only ever adds time. Of
fifteen rounds the cheapest is the one that came closest to having the card to
itself, so it is the best available estimate of what the operator costs; every
other round is that plus somebody else's work. The mean averages the
interference in, and the median keeps half of it.

The first version of this module used the median. Its own validation run is what
argued it down: under live training, medians sat up to 1.6x above the cheapest
round, while the minima across channel widths were clean and monotonic for both
operators. The median is still computed, because the ratio between it and the
minimum is the useful contention signal -- near 1 means the card was quiet,
large means most rounds carried someone else's work -- but the reported cost is
the minimum.

WHY IT CHECKS ITSELF

A measurement method that cannot be wrong in a visible way is not one to trust
after the last two attempts. So the harness times one configuration twice, as
two separately constructed modules of identical shape, and compares what each
reports. They are measuring the same computation; if they disagree, the method
is not working on this machine today and nothing measured with it should be
published.
Run it standalone before believing any operator timing:

    LEAD_RUNTIME_TYPE_CHECKING=false python scripts/common/interleaved_timing.py --compile
"""

import argparse
import pathlib
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lead.config import load_lead_config  # noqa: E402
from lead.policy.transfuser.encoder.deformable_attention import (  # noqa: E402
    MultiScaleDeformableAttention,
)
from lead.policy.transfuser.encoder.transfuser_backbone import (  # noqa: E402
    SelfAttention,
)

# Two independent measurements of the same configuration further apart than this
# mean the method is not working here. This is the gate that matters: it asks
# directly whether the number reproduces.
_REPLICA_TOLERANCE = 0.10
# The median may sit above the quietest round -- that is contention, and the
# minimum is chosen precisely to see past it. Beyond this factor, though, so
# little of the session was quiet that even the minimum is probably somebody
# else's work included.
_INFLATION_LIMIT = 4.0


class Timing:
    """One configuration's cost, and how contended the card was while measuring.

    ``best`` is the reported cost: the cheapest round, which is the one that
    came nearest to having the card alone. ``inflation`` is the median over that
    minimum, and it is the contention signal -- 1.0 means every round was as
    quiet as the quietest, and a large value means most rounds carried other
    work.
    """

    def __init__(self, samples: list[float]) -> None:
        """Summarise the per-round means.

        Args:
            samples: One mean latency per round, in milliseconds.
        """
        ordered = sorted(samples)
        self.samples = ordered
        self.best = ordered[0]
        self.median = statistics.median(ordered)
        self.high = ordered[-1]
        self.inflation = self.median / self.best if self.best > 0 else float("inf")

    def __repr__(self) -> str:
        """Readable form for error messages."""
        return f"{self.best:.3f} ms (median {self.median:.3f}, {self.inflation:.1f}x)"


def _time_once(module, x, iterations: int) -> float:
    """Mean milliseconds per forward over one short burst.

    The synchronise before and after is what makes this a measurement rather
    than a record of how fast Python queues kernels.

    Args:
        module: Module to time, on the device and in eval mode.
        x: Its input.
        iterations: Forwards per burst.

    Returns:
        Mean milliseconds per forward.
    """
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(iterations):
            module(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def time_interleaved(configs, rounds=15, iterations=8, warmup=20):
    """Time every configuration in rotating rounds, cheapest round reported.

    Args:
        configs: ``{name: (module, input_tensor)}``. Modules should already be
            on the device, in eval mode, and compiled if they are going to be.
        rounds: How many times to visit every configuration.
        iterations: Forwards per visit. Small on purpose -- a long visit is a
            contiguous window again, which is the thing being avoided.
        warmup: Forwards run once per configuration before any timing, to pay
            for compilation and autotuning outside the measurement.

    Returns:
        ``{name: Timing}``.
    """
    names = list(configs)
    for name in names:
        module, x = configs[name]
        _time_once(module, x, warmup)

    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(rounds):
        # Rotate, so no configuration is always the one that pays for whatever
        # the start of a round costs.
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            module, x = configs[name]
            samples[name].append(_time_once(module, x, iterations))
    return {name: Timing(values) for name, values in samples.items()}


def check_replica(results, name_a, name_b, tolerance=_REPLICA_TOLERANCE):
    """Refuse the whole measurement if two copies of one thing disagree.

    Args:
        results: Output of :func:`time_interleaved`.
        name_a: One configuration.
        name_b: A separately built module of identical shape.
        tolerance: Allowed relative gap between the two reported costs.

    Raises:
        SystemExit: If they disagree by more than the tolerance, or either was
            measured on a card too busy for even its quietest round to mean
            anything.
    """
    a, b = results[name_a], results[name_b]
    for name, timing in ((name_a, a), (name_b, b)):
        if timing.inflation > _INFLATION_LIMIT:
            raise SystemExit(
                f"{name}: the median round is {timing.inflation:.1f}x the "
                f"cheapest ({timing.best:.3f} against {timing.median:.3f} ms). "
                f"Almost no round ran on a quiet card, so even the minimum is "
                f"likely carrying someone else's work.",
            )
    gap = abs(a.best - b.best) / min(a.best, b.best)
    if gap > tolerance:
        raise SystemExit(
            f"two identical configurations measured {a} and {b}, a "
            f"{100 * gap:.0f}% disagreement. They are the same computation, so "
            f"this is the method failing rather than a difference between "
            f"operators. Nothing measured in this session should be published.",
        )
    return gap


def build_probe(kind, tokens, channels, image_grid, bev_grid, config, batch, points, device, compile_):
    """One operator instance and an input for it.

    Args:
        kind: ``"dense"`` or ``"deformable"``.
        tokens: Sequence length.
        channels: Embedding width.
        image_grid: ``(rows, cols)`` of the image side.
        bev_grid: ``(rows, cols)`` of the BEV side.
        config: The transfuser config, for the head count.
        batch: Batch size.
        points: Sampled points per query for the sparse operator.
        device: Device to build on.
        compile_: Whether to wrap in ``torch.compile``.

    Returns:
        ``(module, input_tensor)``.
    """
    x = torch.randn(batch, tokens, channels, device=device)
    if kind == "dense":
        module = SelfAttention(channels, config.n_head, 0.0, 0.0)
    else:
        module = MultiScaleDeformableAttention(
            n_embd=channels,
            n_head=config.n_head,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            spatial_shapes=(image_grid, bev_grid),
            num_points=points,
        )
    module = module.to(device).eval()
    if compile_:
        module = torch.compile(module)
    return module, x


def main() -> int:
    """Say whether operator timing is reproducible on this machine right now.

    Returns:
        Process exit code; non-zero means do not trust timings taken here today.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    if arguments.device != "cuda":
        print("CUDA events are the only timer here; run this on the GPU.")
        return 1

    device = torch.device(arguments.device)
    config = load_lead_config().policy.transfuser

    # The four channel widths the four fusion blocks actually run at, at the
    # shipped token count. Monotonicity in channels is a property the real
    # operators have, so a violation is a measurement fault.
    widths = (64, 128, 256, 512)
    grids = ((12, 36), (10, 12))
    configs = {}
    for channels in widths:
        for kind in ("dense", "deformable"):
            configs[f"{kind}_{channels}"] = build_probe(
                kind, 552, channels, *grids, config, arguments.batch_size,
                arguments.num_points, device, arguments.compile,
            )
    # The replica: separately constructed, identical shape. Measuring it apart
    # from its twin is the only check here that cannot be argued with.
    configs["dense_256_replica"] = build_probe(
        "dense", 552, 256, *grids, config, arguments.batch_size,
        arguments.num_points, device, arguments.compile,
    )

    print(
        f"\ndevice={device}  batch={arguments.batch_size}  "
        f"rounds={arguments.rounds}  iterations/round={arguments.iterations}  "
        f"{'compiled' if arguments.compile else 'eager'}",
    )
    print(f"{len(configs)} configurations, interleaved and rotated\n")

    results = time_interleaved(
        configs, arguments.rounds, arguments.iterations, arguments.warmup,
    )

    header = (
        f"  {'configuration':<24}{'cost ms':>10}{'median':>10}{'worst':>10}"
        f"{'inflation':>11}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in configs:
        timing = results[name]
        print(
            f"  {name:<24}{timing.best:>10.3f}{timing.median:>10.3f}"
            f"{timing.high:>10.3f}{timing.inflation:>10.1f}x",
        )

    print("\n  --- does the method work here? ---")
    gap = check_replica(results, "dense_256", "dense_256_replica")
    print(
        f"  Two separately built copies of the same configuration agree to "
        f"{100 * gap:.1f}%.",
    )

    worst = max(results.values(), key=lambda t: t.inflation)
    print(
        f"  Most contended configuration: its median round was "
        f"{worst.inflation:.1f}x its cheapest.",
    )

    # Monotonicity is a second, independent check: same tokens, wider operator,
    # so the cost cannot fall.
    for kind in ("dense", "deformable"):
        series = [(c, results[f"{kind}_{c}"].best) for c in widths]
        for (narrow, narrow_ms), (wide, wide_ms) in zip(series, series[1:], strict=False):
            if narrow_ms > wide_ms * 1.25:
                raise SystemExit(
                    f"  {kind}: {narrow} channels measured {narrow_ms:.3f} ms and "
                    f"{wide} measured {wide_ms:.3f} ms. A wider operator cannot be "
                    f"cheaper, so even the quietest rounds are contended.",
                )
    print("  Both operators are monotonic in channel width, as they must be.")
    print(
        "\n  Timings taken this way, in this session, are usable. Re-run this "
        "before any\n  measurement you intend to publish -- it answers a "
        "question about the machine\n  right now, not about the code.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
