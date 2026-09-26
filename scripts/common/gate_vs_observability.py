"""Test whether the gate's learned bias behaves the way the formulation says.

Chapter three derives that a softmax with an additive logit bias reproduces
inverse-variance weighting exactly when the bias is the log precision of the
modality. That derivation justifies the gate's functional form, but it says
nothing about whether a gate trained by supervision actually lands there. This
measures it.

The prediction is stated as a difference rather than a level, because the
softmax is shift invariant: only ``b_cam - b_lid`` has any effect, so only that
difference can be predicted. If observability stands in for precision, then

    b_cam - b_lid  ==  log( V_cam / V_lid )  + constant

over the tokens where both modalities are supervised. Two things are reported:
the correlation of the two sides, which asks whether the gate tracks
observability at all, and the slope of a least-squares fit, which asks whether
it tracks it at the rate the derivation predicts. A slope near one supports the
formulation; a slope near zero means the gate learned something else.

Nothing here degrades the input by default: the question is what the gate
learned, not how it reacts. A degraded condition is run as well, because the
gate is supposed to matter most when a sensor is failing.
"""

import argparse
import csv
import pathlib
import sys

import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "common"))

from analyze_gate import FusionProbe, load_model, to_device  # noqa: E402

from lead.policy.transfuser.utils.sensor_degradation import degrade_batch  # noqa: E402

_DAMAGE_SEED = 20260820
# Tokens whose observability is essentially zero in either modality carry no
# information about a ratio and would dominate a log through the epsilon.
_MIN_OBSERVABILITY = 0.05
_CAMERA, _LIDAR = 0, 1


def statistics(x: torch.Tensor, y: torch.Tensor) -> dict:
    """Correlation and least-squares slope of y on x.

    Args:
        x: Predictor, one dimensional.
        y: Response, one dimensional.

    Returns:
        Count, correlation, slope and intercept.
    """
    n = x.numel()
    if n < 2:
        return {"n": n, "r": float("nan"), "slope": float("nan"),
                "intercept": float("nan")}
    x_centred = x - x.mean()
    y_centred = y - y.mean()
    denominator = (x_centred * x_centred).sum()
    slope = (x_centred * y_centred).sum() / denominator.clamp(min=1e-12)
    correlation = (x_centred * y_centred).sum() / (
        x_centred.norm() * y_centred.norm()
    ).clamp(min=1e-12)
    return {
        "n": int(n),
        "r": float(correlation),
        "slope": float(slope),
        "intercept": float(y.mean() - slope * x.mean()),
    }


def collect(model, lead_config, loader, condition, batches, device) -> dict:
    """Gather paired (log observability ratio, gate bias difference) samples.

    Args:
        model: Loaded policy in eval mode.
        lead_config: Config it was trained with.
        loader: Loader over the probe frames.
        condition: ``(modality, severity)``.
        batches: How many batches to run.
        device: Device to run on.

    Returns:
        The statistics of the fit, plus how many tokens entered it.
    """
    modality, severity = condition
    # The policy the runner hands back may wrap the transfuser, so the module
    # is looked up rather than assumed to sit at the top level.
    token_targets = getattr(model, "observability_token_targets", None)
    if token_targets is None:
        for module in model.modules():
            if type(module).__name__ == "ObservabilityTokenTargets":
                token_targets = module
                break
    if token_targets is None:
        return {"n": 0, "r": float("nan"), "slope": float("nan"),
                "intercept": float("nan"), "note": "no token targets module"}
    probe = FusionProbe(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(_DAMAGE_SEED)

    predictor: list[torch.Tensor] = []
    response: list[torch.Tensor] = []
    seen = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                if seen >= batches:
                    break
                batch = to_device(batch, device)
                if severity > 0:
                    batch = degrade_batch(batch, modality, severity, generator)
                    # The inference-time degradation damages the input but
                    # leaves the label alone, which is right for evaluation
                    # and wrong here: the prediction is about the precision
                    # that actually holds, so the target has to fall with the
                    # sensor exactly as the training curriculum makes it fall.
                    channel = _CAMERA if modality == "camera" else _LIDAR
                    batch["observability"] = batch["observability"].clone()
                    batch["observability"][:, channel] *= (1.0 - severity)
                with autocast(
                    device_type="cuda",
                    dtype=lead_config.training.optimization.torch_dtype,
                    enabled=(
                        lead_config.training.optimization.use_mixed_precision_training
                    ),
                ):
                    model(batch)

                gate_logits = probe._gate_logits  # noqa: SLF001
                if not gate_logits:
                    return {"n": 0, "r": float("nan"), "slope": float("nan"),
                            "intercept": float("nan"), "note": "no gate"}
                # Average the blocks: they all bias the same token grid, and
                # the derivation is about the operator rather than one layer.
                bias = torch.stack(
                    [value.float() for value in gate_logits.values()],
                ).mean(0)

                targets, mask = token_targets(
                    batch["observability"].float(),
                    batch["observability_mask"].float(),
                )
                targets = targets.float()
                mask = mask.float()

                usable = (
                    (mask[..., _CAMERA] > 0)
                    & (mask[..., _LIDAR] > 0)
                    & (targets[..., _CAMERA] > _MIN_OBSERVABILITY)
                    & (targets[..., _LIDAR] > _MIN_OBSERVABILITY)
                )
                if not bool(usable.any()):
                    seen += 1
                    continue
                log_ratio = (
                    targets[..., _CAMERA].log() - targets[..., _LIDAR].log()
                )[usable]
                bias_difference = (
                    bias[..., _CAMERA] - bias[..., _LIDAR]
                )[usable]
                predictor.append(log_ratio.cpu())
                response.append(bias_difference.cpu())
                seen += 1
    finally:
        probe.close()

    if not predictor:
        return {"n": 0, "r": float("nan"), "slope": float("nan"),
                "intercept": float("nan"), "note": "no supervised tokens"}
    x = torch.cat(predictor)
    y = torch.cat(response)
    result = statistics(x, y)
    result["bias_sd"] = float(y.std())
    result["logratio_sd"] = float(x.std())
    return result


def main() -> None:
    """Run the test for each model and condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", default=["none:0"])
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=pathlib.Path,
                        default=ROOT / "results/gate_vs_observability.csv")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    rows = []
    for entry in args.models:
        name, _, path = entry.partition("=")
        print(f"\n=== {name} ===", flush=True)
        lead_config, model = load_model(pathlib.Path(path), device)
        model.eval()
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
        for pair in args.conditions:
            modality, _, severity = pair.partition(":")
            result = collect(
                model, lead_config, loader,
                (modality, float(severity)), args.batches, device,
            )
            result |= {"model": name, "condition": pair}
            rows.append(result)
            print(
                f"  {pair:12} n={result['n']:>7}  r={result['r']:+.4f}  "
                f"slope={result['slope']:+.4f}  "
                f"sd(bias diff)={result.get('bias_sd', float('nan')):.4f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "condition", "n", "r", "slope", "intercept",
              "bias_sd", "logratio_sd", "note"]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
