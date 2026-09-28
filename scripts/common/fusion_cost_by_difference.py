"""What fusion costs, measured by removing it rather than by watching it.

WHY NOT JUST PROFILE IT

`forward_profile.py` puts CUDA events around each module and reports what falls
between them. Asked three times for one number -- the fusion blocks' share of
rung0's forward pass -- it answered 2.2%, 12.9% and 6.9%, while the whole
forward pass stayed within a millisecond of itself across the same runs. A
stable total with unstable parts is an attribution failure, not a noisy machine.

The cause is not compilation: `PolicyRunner` loads the policy eager, so the
module boundaries the hooks attach to are real. It is that this GPU runs another
user's process alongside ours. When the driver time-slices away from our
context, our stream stalls, and the stall is charged to whichever module's event
window happened to be open. The total absorbs every stall wherever it lands; the
parts get whichever ones fell inside them.

MEASURING BY DIFFERENCE INSTEAD

Run the whole model, then run it again with fusion replaced by a pass-through,
and subtract. The difference is what fusion costs, and no part of it depends on
attributing time to a module: both measurements are of the entire forward pass,
which is the quantity that was stable all along.

The pass-through returns `fuse_features`' inputs unchanged, which skips the
pooling, the transformer, the interpolation back up and the residual add -- the
whole block. The model's outputs become meaningless. That is fine; nothing here
reads them.

The measurements themselves use interleaved_timing, so each is the cheapest of
several rotating rounds rather than a mean over one contiguous window. The
modified and unmodified models are interleaved with *each other*, which needs
the pass-through installed and removed around every individual call rather than
around a whole run. The first version of this script did the latter, and rung0
came back unresolved with an 84 ms noise floor against a 96 ms difference.

WHAT IT REFUSES TO DO

Subtracting two noisy numbers gives a noisier one. If fusion is a few percent of
the forward pass and two measurements of the *same* model disagree by more than
that, the difference is inside the noise and means nothing. So a replica of the
unmodified model is timed alongside, and the script refuses to report a fusion
cost it cannot resolve above the disagreement between two measurements of one
thing. That refusal is the answer in that case: not "fusion is small" but "this
machine cannot tell you how small".

That check alone is not enough, and finding out cost two runs. One of them
passed it comfortably -- a noise floor of 0.37 ms against a 10.71 ms difference,
twenty-nine times clear -- and the next run minutes later, with identical
settings, produced a difference 49% smaller. The intact model and its replica
had both happened to catch quiet slots, so they agreed, and their agreement
certified a session in which the ablated model had not been so lucky. Two
configurations agreeing does not vouch for a third.

So the whole measurement repeats, and the differences from the repeats have to
agree with each other. That is the check that cannot be passed by luck: chance
does not favour the same configuration twice in a row.
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
from interleaved_timing import time_interleaved  # noqa: E402


class ForwardUnderAutocast:
    """The policy's forward pass, in the dtype it was trained in.

    A plain call fails: the weights are fp32 and the batch arrives in the
    training dtype, so the first convolution rejects the pair.
    """

    def __init__(self, model, lead_config) -> None:
        """Bind a model to its training precision.

        Args:
            model: The loaded policy.
            lead_config: Its config, for the autocast dtype.
        """
        self.model = model
        optimization = lead_config.training.optimization
        self.dtype = optimization.torch_dtype
        self.enabled = optimization.use_mixed_precision_training

    def __call__(self, batch):
        """Run one forward pass.

        Args:
            batch: An already-on-device batch.

        Returns:
            Whatever the policy returns; nothing here reads it.
        """
        with autocast(device_type="cuda", dtype=self.dtype, enabled=self.enabled):
            return self.model(batch)


class ForwardWithoutFusion(ForwardUnderAutocast):
    """The same forward pass with the fusion blocks skipped.

    The pass-through is installed and removed around each individual call
    rather than around a whole measurement run. That is what lets the modified
    and unmodified models be interleaved with each other: if one had to hold the
    patch for its entire run, the two would be separate contiguous windows
    again, and the difference between them would carry whatever the card was
    doing during each -- which is exactly how the first version of this script
    failed to resolve rung0, reporting a noise floor of 84 ms against a
    difference of 96 ms.

    Assigning to the instance shadows the class method, so the replacement is
    called without ``self``. Deleting the attribute uncovers the real method.
    The two attribute operations sit inside the timed region and cost
    nanoseconds against a forward pass of a hundred milliseconds.
    """

    def __init__(self, model, lead_config) -> None:
        """Bind to a model whose backbone fuses.

        Args:
            model: The loaded policy.
            lead_config: Its config, for the autocast dtype.

        Raises:
            SystemExit: If the backbone has no ``fuse_features``, so there is
                nothing to remove and the difference would measure nothing.
        """
        super().__init__(model, lead_config)
        self.backbone = getattr(model, "backbone", None)
        if not hasattr(self.backbone, "fuse_features"):
            raise SystemExit(
                "this backbone has no fuse_features, so fusion cannot be "
                "removed by replacing it; the difference below would measure "
                "nothing.",
            )

    def __call__(self, batch):
        """Run one forward pass with fusion skipped.

        Args:
            batch: An already-on-device batch.

        Returns:
            Whatever the policy returns; nothing here reads it.
        """
        self.backbone.fuse_features = lambda image, lidar, layer_idx: (image, lidar)
        try:
            return super().__call__(batch)
        finally:
            del self.backbone.fuse_features


def measure_one(name, model, lead_config, batch, arguments):
    """Time the model with fusion, without it, and with it again, interleaved.

    All three go into one interleaved pass, so a burst of someone else's work
    lands across them rather than on whichever was scheduled against it. The
    third is not redundant: it is the same computation as the first, so the gap
    between them is this machine's noise floor right now, and the difference
    being sought has to clear it to mean anything.

    Args:
        name: Label for the report.
        model: The loaded policy.
        lead_config: Its config.
        batch: An already-on-device batch.
        arguments: Parsed arguments.

    Returns:
        ``(full, ablated, replica)`` timings.
    """
    results = time_interleaved(
        {
            f"{name}_full": (ForwardUnderAutocast(model, lead_config), batch),
            f"{name}_ablated": (ForwardWithoutFusion(model, lead_config), batch),
            f"{name}_replica": (ForwardUnderAutocast(model, lead_config), batch),
        },
        arguments.rounds,
        arguments.iterations,
        arguments.warmup,
    )
    return (
        results[f"{name}_full"],
        results[f"{name}_ablated"],
        results[f"{name}_replica"],
    )


def report(name, full, ablated, replica) -> bool:
    """Print one model's result and say whether it resolved.

    Args:
        name: Label.
        full: Timing of the unmodified model.
        ablated: Timing with fusion removed.
        replica: Timing of the unmodified model, measured again.

    Returns:
        True if the fusion cost cleared the noise floor.
    """
    difference = full.best - ablated.best
    floor = abs(full.best - replica.best)
    share = difference / full.best if full.best > 0 else 0.0

    print(f"\n  {name}")
    print(f"    whole forward, with fusion   {full.best:8.2f} ms")
    print(f"    whole forward, fusion removed{ablated.best:8.2f} ms")
    print(f"    difference                   {difference:8.2f} ms   ({100 * share:.1f}%)")
    print(f"    noise floor (two runs of the same model)  {floor:.2f} ms")

    if difference <= 0:
        print(
            "    UNRESOLVED: removing fusion did not make the model faster. "
            "Either the pass-through\n    did not take effect, or the "
            "measurement cannot see a difference this small.",
        )
        return False
    if difference < floor * 2:
        print(
            f"    UNRESOLVED: the difference is {difference / floor:.1f}x the "
            f"noise floor, which is not enough\n    to call it. What this says "
            f"is that fusion costs less than this machine can resolve,\n    not "
            f"that it costs nothing.",
        )
        return False
    print(
        f"    RESOLVED: {difference / floor:.0f}x the noise floor. Fusion is "
        f"{100 * share:.1f}% of the forward pass.",
    )
    return True


def main() -> int:
    """Measure fusion's cost by difference for each named checkpoint.

    Returns:
        Process exit code; non-zero if no model resolved.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    # Many short rounds rather than few long ones. A round of three forwards is
    # 200 ms, long enough that a continuously busy card contends with almost
    # every one of them; at one forward per round the minimum has sixty chances
    # to catch a quiet slot. Moving from 12x3 to 60x1 took rung0's noise floor
    # from 16.44 ms to 0.37 ms and turned both models from unresolved to
    # resolved.
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    # Two independent repeats of the whole measurement. One run's internal
    # check can pass by luck; two runs agreeing cannot.
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--agreement", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    if arguments.device != "cuda":
        print("CUDA events are the only timer here; run this on the GPU.")
        return 1

    device = torch.device(arguments.device)
    print(
        f"\nfusion cost by difference  batch={arguments.batch_size}  "
        f"rounds={arguments.rounds}  iterations/round={arguments.iterations}",
    )
    print(
        "the reported cost of each configuration is its cheapest round; see "
        "interleaved_timing.py",
    )

    resolved = 0
    for pair in arguments.models:
        name, _, path = pair.partition("=")
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        dataset = model.build_dataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=arguments.workers,
        )
        batch = to_device(next(iter(loader)), device)

        shares = []
        passed = True
        for repeat in range(arguments.repeats):
            full, ablated, replica = measure_one(
                f"{name}_r{repeat}", model, lead_config, batch, arguments,
            )
            label = name if arguments.repeats == 1 else f"{name}, repeat {repeat + 1}"
            if report(label, full, ablated, replica):
                shares.append((full.best - ablated.best) / full.best)
            else:
                passed = False

        if arguments.repeats > 1:
            if len(shares) < arguments.repeats:
                print(
                    "\n"
                    f"  {name}: only {len(shares)} of {arguments.repeats} "
                    f"repeats resolved, so the card was not quiet enough for "
                    f"this measurement.",
                )
                passed = False
            else:
                spread = (max(shares) - min(shares)) / min(shares)
                print(
                    "\n"
                    f"  {name}: repeats gave "
                    f"{', '.join(f'{100 * s:.1f}%' for s in shares)}, spread "
                    f"{100 * spread:.0f}%.",
                )
                if spread > arguments.agreement:
                    print(
                        f"  REJECTED: repeats of one measurement should agree "
                        f"within {100 * arguments.agreement:.0f}%. They do not, "
                        f"so no single run of this is publishable however clean "
                        f"its own internal check looked.",
                    )
                    passed = False
                else:
                    print(
                        f"  ACCEPTED: fusion is "
                        f"{100 * sum(shares) / len(shares):.1f}% of the forward "
                        f"pass, and it reproduces.",
                    )
        resolved += passed

    print(
        "\n  A difference of whole-model timings needs no attribution, which is "
        "the point: the\n  quantity that was stable across every earlier run is "
        "the whole forward pass, and this\n  asks only that question twice.",
    )
    return 0 if resolved else 2


if __name__ == "__main__":
    sys.exit(main())
