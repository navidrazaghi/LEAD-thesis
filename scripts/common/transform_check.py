"""Ask whether the gate failed to learn, or learned something and was read wrong.

The alignment measurement compared the raw gate bias against the log ratio of
observability and found a correlation of at most 0.30. That was read as the
gate failing to learn the relation the formulation prescribes. There is a
second reading that has to be ruled out first.

The gate is supervised with binary cross entropy against the observability
target, so training pushes ``sigmoid(b)`` toward ``V``: the gate's output is a
*logit* of observability. Equation (3-13) asks for ``log V`` as the bias. Those
are different functions of the same quantity -- ``logit`` is unbounded above,
``log`` saturates at zero -- so a gate that learned its supervision perfectly
would still correlate imperfectly with ``log V``.

Three comparisons separate the two readings:

  A  raw bias        vs  log ratio      what the earlier measurement did
  B  raw bias        vs  logit ratio    did the gate learn its own supervision?
  C  log-sigmoid bias vs log ratio      would the corrected bias satisfy (3-13)?

If B is strong and A is weak, the gate learned correctly and the architecture
reads it through the wrong transform, which is a one-line fix. If B is also
weak, the gate genuinely did not learn and the fix is a training change.
"""

import argparse
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import FusionProbe, load_model, to_device  # noqa: E402

_MIN_OBSERVABILITY = 0.05
_MAX_OBSERVABILITY = 0.95  # logit blows up as V approaches one
_CAMERA, _LIDAR = 0, 1


def correlate(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Pearson correlation and least-squares slope of y on x."""
    if x.numel() < 2:
        return float("nan"), float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    slope = (xc * yc).sum() / (xc * xc).sum().clamp(min=1e-12)
    r = (xc * yc).sum() / (xc.norm() * yc.norm()).clamp(min=1e-12)
    return float(r), float(slope)


def main() -> None:
    """Run the three comparisons for each model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    for entry in args.models:
        name, _, path = entry.partition("=")
        print(f"\n=== {name} ===", flush=True)
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
        targets_module = None
        for module in model.modules():
            if type(module).__name__ == "ObservabilityTokenTargets":
                targets_module = module
                break
        if targets_module is None:
            print("  no token targets module")
            continue

        dataset = model.build_dataset()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
        )
        probe = FusionProbe(model)
        raw, logv, logitv, logsig = [], [], [], []
        seen = 0
        try:
            with torch.inference_mode():
                for batch in loader:
                    if seen >= args.batches:
                        break
                    batch = to_device(batch, device)
                    with autocast(
                        device_type="cuda",
                        dtype=lead_config.training.optimization.torch_dtype,
                        enabled=(
                            lead_config.training.optimization.use_mixed_precision_training
                        ),
                    ):
                        model(batch)
                    gate = probe._gate_logits  # noqa: SLF001
                    if not gate:
                        print("  no gate")
                        break
                    bias = torch.stack([v.float() for v in gate.values()]).mean(0)
                    t, m = targets_module(
                        batch["observability"].float(),
                        batch["observability_mask"].float(),
                    )
                    t, m = t.float(), m.float()
                    usable = (
                        (m[..., _CAMERA] > 0)
                        & (m[..., _LIDAR] > 0)
                        & (t[..., _CAMERA] > _MIN_OBSERVABILITY)
                        & (t[..., _LIDAR] > _MIN_OBSERVABILITY)
                        & (t[..., _CAMERA] < _MAX_OBSERVABILITY)
                        & (t[..., _LIDAR] < _MAX_OBSERVABILITY)
                    )
                    if not bool(usable.any()):
                        seen += 1
                        continue
                    vc, vl = t[..., _CAMERA][usable], t[..., _LIDAR][usable]
                    bc, bl = bias[..., _CAMERA][usable], bias[..., _LIDAR][usable]
                    raw.append((bc - bl).cpu())
                    logv.append((vc.log() - vl.log()).cpu())
                    logitv.append(((vc / (1 - vc)).log() - (vl / (1 - vl)).log()).cpu())
                    logsig.append(
                        (
                            torch.nn.functional.logsigmoid(bc)
                            - torch.nn.functional.logsigmoid(bl)
                        ).cpu()
                    )
                    seen += 1
        finally:
            probe.close()

        if not raw:
            print("  nothing usable")
            continue
        raw = torch.cat(raw)
        logv = torch.cat(logv)
        logitv = torch.cat(logitv)
        logsig = torch.cat(logsig)
        print(f"  tokens: {raw.numel()}")
        for label, x, y in (
            ("A  raw bias      vs  log V ratio    ", logv, raw),
            ("B  raw bias      vs  logit V ratio  ", logitv, raw),
            ("C  log-sigmoid b vs  log V ratio    ", logv, logsig),
        ):
            r, s = correlate(x, y)
            print(f"  {label} r={r:+.4f}  slope={s:+.4f}")


if __name__ == "__main__":
    main()
