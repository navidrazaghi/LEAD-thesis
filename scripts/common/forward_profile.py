"""Where the forward pass actually spends its time.

The thesis argues the deformable operator is linear where dense attention is
quadratic, then reports that at this architecture's 552 tokens the asymptotic
advantage does not become a wall-clock gain -- measured, the deformable rung is
13% slower. That leaves an obvious question the cost table does not answer: if
attention is not where the time goes, where does it go?

The answer bounds every future attempt at making fusion cheaper. Optimising a
component that is a tenth of the forward pass cannot return more than a tenth,
whatever the operator, and knowing that number before spending months on an
operator is worth the few minutes this takes.

Timing is per module, with CUDA events around each one and a synchronise before
reading them, because CUDA calls return before the work is done and timing
without the barrier measures how fast Python can queue kernels.

Two levels are reported: the policy's own children -- backbone, the auxiliary
heads, the planning decoder -- and inside the backbone, its encoders and the
fusion transformer, which is where the attention lives.

Read the percentages, not the milliseconds, if anything else is on the GPU.
Contention inflates the absolute numbers; the shares are what this is for. The
absolute figure measured on an idle device is in results/cost.csv.
"""

import argparse
import collections
import pathlib
import statistics
import sys

import torch
from torch.amp.autocast_mode import autocast

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import load_model, to_device  # noqa: E402


class ModuleTimer:
    """Records the wall time of every module it is attached to.

    CUDA events rather than a host clock: the host returns as soon as the work
    is queued, so a host timer around a module measures the queueing and not
    the work.
    """

    def __init__(self) -> None:
        """Start with nothing timed and no handles held."""
        self.totals: dict[str, list[float]] = collections.defaultdict(list)
        self._events: dict[str, tuple] = {}
        self._handles: list = []

    def watch(self, name: str, module: torch.nn.Module) -> None:
        """Time one module.

        Args:
            name: The label to report it under.
            module: The module to time.
        """

        def pre(_m, _i, name=name):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._events[name] = (start, end)

        def post(_m, _i, output, name=name):
            pair = self._events.pop(name, None)
            if pair is None:
                return output
            start, end = pair
            end.record()
            self.totals[name].append((start, end))
            return output

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def collect(self) -> dict[str, float]:
        """Resolve every recorded event pair into milliseconds.

        Returns:
            Mean milliseconds per call, by module name.
        """
        torch.cuda.synchronize()
        resolved = {}
        for name, pairs in self.totals.items():
            times = [start.elapsed_time(end) for start, end in pairs]
            if times:
                resolved[name] = statistics.mean(times)
        return resolved

    def reset(self) -> None:
        """Forget the warmup's measurements."""
        self.totals.clear()
        self._events.clear()

    def remove(self) -> None:
        """Detach every hook, so a second model is not timed through this one."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def profile(model, lead_config, loader, warmup, repeats, device):
    """Time the policy's children and the backbone's, over one batch.

    Args:
        model: The loaded policy.
        lead_config: Its config.
        loader: Loader over the probe frames.
        warmup: Iterations to discard before measuring.
        repeats: Iterations to measure.
        device: Device to run on.

    Returns:
        The per-module means, and the mean of the whole forward pass.
    """
    batch = to_device(next(iter(loader)), device)
    model.eval()

    timer = ModuleTimer()
    for name, child in model.named_children():
        timer.watch(name, child)

    # The backbone's own children are not where its time goes. It iterates the
    # encoders stage by stage and fuses between them, so the encoder modules'
    # forward is never called and a hook on them never fires -- which is how a
    # first pass at this attributed under a millisecond to a backbone taking
    # sixty. The stages and the fusion blocks have to be hooked directly.
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        for name, child in backbone.named_children():
            timer.watch(f"backbone.{name}", child)
        for encoder in ("image_encoder", "lidar_encoder"):
            module = getattr(backbone, encoder, None)
            if module is None:
                continue
            for stage, layer in module.items():
                timer.watch(f"{encoder}.{stage}", layer)
        # Dense fusion keeps its blocks in `transformers`; the deformable
        # backbone keeps them in `blocks`, and both are worth naming as one
        # line so the two are comparable.
        for attribute in ("transformers", "blocks"):
            stack = getattr(backbone, attribute, None)
            if stack is None:
                continue
            for index, block in enumerate(stack):
                timer.watch(f"fusion.{attribute}[{index}]", block)

    optimization = lead_config.training.optimization
    whole = []
    with torch.inference_mode():
        for index in range(warmup + repeats):
            if index == warmup:
                timer.reset()
                whole.clear()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast(
                device_type="cuda",
                dtype=optimization.torch_dtype,
                enabled=optimization.use_mixed_precision_training,
            ):
                model(batch)
            end.record()
            whole.append((start, end))

    torch.cuda.synchronize()
    per_module = timer.collect()
    timer.remove()
    total = statistics.mean(start.elapsed_time(end) for start, end in whole)
    return per_module, total


def main() -> int:
    """Profile each named checkpoint and print the breakdown.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    if arguments.device != "cuda":
        print("CUDA events are the only timer here; run this on the GPU.")
        return 1

    device = torch.device(arguments.device)
    for pair in arguments.models:
        name, _, path = pair.partition("=")
        lead_config, model = load_model(pathlib.Path(path), device)
        dataset = model.build_dataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=arguments.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=arguments.workers,
        )
        per_module, total = profile(
            model,
            lead_config,
            loader,
            arguments.warmup,
            arguments.repeats,
            device,
        )

        print(f"\n{name}  whole forward {total:.2f} ms  (batch {arguments.batch_size})")
        print(f"  {'module':<34}{'ms':>9}{'share':>9}")

        top = {k: v for k, v in per_module.items() if "." not in k}
        fusion = {k: v for k, v in per_module.items() if k.startswith("fusion.")}
        encoders = {
            k: v for k, v in per_module.items() if k.startswith(("image_encoder.", "lidar_encoder."))
        }
        print("  --- policy children ---")
        for key, value in sorted(top.items(), key=lambda kv: -kv[1]):
            print(f"  {key:<34}{value:9.2f}{100 * value / total:8.1f}%")

        if encoders:
            print("  --- encoder stages, inside the backbone ---")
            for key, value in sorted(encoders.items(), key=lambda kv: -kv[1]):
                print(f"  {key:<34}{value:9.2f}{100 * value / total:8.1f}%")

        if fusion:
            print("  --- fusion blocks, inside the backbone ---")
            for key, value in sorted(fusion.items(), key=lambda kv: -kv[1]):
                print(f"  {key:<34}{value:9.2f}{100 * value / total:8.1f}%")

        # The number the whole exercise is for: what an optimised fusion
        # operator could return at most, whatever the operator.
        fusion_total = sum(fusion.values())
        encoder_total = sum(encoders.values())
        print("  --- what this answers ---")
        print(f"  {'all fusion blocks':<34}{fusion_total:9.2f}"
              f"{100 * fusion_total / total:8.1f}%   <- the ceiling on any"
              " attention optimisation")
        print(f"  {'all encoder stages':<34}{encoder_total:9.2f}"
              f"{100 * encoder_total / total:8.1f}%")
        accounted = sum(top.values())
        print(f"  {'policy children together':<34}{accounted:9.2f}"
              f"{100 * accounted / total:8.1f}%")
    print(
        "\nShares, not milliseconds, are what this measures if anything else is "
        "on the GPU.\nThe idle-device absolute is in results/cost.csv.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
