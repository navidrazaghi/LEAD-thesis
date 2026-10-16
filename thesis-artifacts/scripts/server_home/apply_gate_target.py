"""Let the gate be supervised on the quantity chapter three actually derives.

The optimal bias is the log of the precision, and the thesis treats
observability as the normalized form of that precision, so the prescribed bias
is the log of observability up to an additive constant. Sigmoid plus binary
cross entropy does not converge there: it drives the gate towards the *log
odds*, which exceeds the log by -log(1 - v). That term is 0.69 at v = 0.5, 2.30
at v = 0.9 and 4.61 at v = 0.99, and since the softmax turns a bias difference
into a mass ratio exponentially, two modalities at 0.9 and 0.5 come out five
times further apart than inverse-variance weighting asks for.

This adds the alternative target. It is off by default, because every result
collected so far was trained under the log-odds form and an evaluation is
queued on this machine that will load this code.
"""

import io
import pathlib
import sys

ROOT = pathlib.Path.home() / "LEAD/lead"
CONFIG = ROOT / "src/lead/config/policy/transfuser/observability_config.py"
GATE = ROOT / "src/lead/policy/transfuser/encoder/observability_gate.py"
POLICY = ROOT / "src/lead/policy/transfuser/transfuser.py"

CONFIG_ANCHOR = """    observability_gate_loss_weight: float = 1.0"""

CONFIG_BLOCK = '''    observability_gate_loss_weight: float = 1.0
    # Which quantity the gate is pulled towards. "logit" is what every result
    # in this project was trained under: sigmoid on the gate output, binary
    # cross entropy against observability, which converges on the log odds.
    # "log" is what the inverse-variance derivation actually prescribes, the
    # log of observability, and differs from the log odds by -log(1 - v) --
    # negligible for a blind modality, 2.3 nats for one at 0.9. Only the
    # difference between the two modalities of a token can matter, since the
    # softmax is shift invariant, so the "log" objective compares centred
    # values and a token with one supervised modality contributes nothing.
    observability_gate_target: str = "logit"'''

GATE_ANCHOR = '''def gate_loss(
    gate_logits: list[jt.Float[torch.Tensor, "B T L"]],
    target: jt.Float[torch.Tensor, "B T L"],
    mask: jt.Float[torch.Tensor, "B T L"],
) -> jt.Float[torch.Tensor, ""]:
    """Masked binary cross entropy of every fusion block\'s gate.

    Args:
        gate_logits: The gate logits of each block, in forward order.
        target: Per-token observability targets.
        mask: Which token/modality pairs carry a measurement.

    Returns:
        The mean loss over the blocks; zero when nothing is supervised.
    """
    supervised = mask.sum().clamp(min=1.0)
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for logits in gate_logits:
        elementwise = F.binary_cross_entropy_with_logits(
            logits.float(),
            target,
            reduction="none",
        )
        total = total + (elementwise * mask).sum() / supervised
    return total / max(len(gate_logits), 1)'''

GATE_BLOCK = '''# Observability of exactly zero is reachable -- a modality that resolves
# nothing -- and its log is not. The floor caps the target at -9.2, which is
# already far past the point where a modality has been shut out of the softmax.
_LOG_TARGET_FLOOR = 1e-4


def _centred_log_gate_loss(
    logits: jt.Float[torch.Tensor, "B T L"],
    target: jt.Float[torch.Tensor, "B T L"],
    mask: jt.Float[torch.Tensor, "B T L"],
) -> jt.Float[torch.Tensor, ""]:
    """Pull the gate towards the log of observability, up to a constant.

    The softmax ignores a bias added equally to every modality, so only the
    difference across the modality axis is defined. Both sides are therefore
    centred on the supervised modalities of each token before they are
    compared, which makes the objective invariant to that constant exactly
    rather than approximately. A token with fewer than two supervised
    modalities has no defined difference and is dropped.

    Args:
        logits: One block\'s gate output.
        target: Per-token observability targets, in [0, 1].
        mask: Which token/modality pairs carry a measurement.

    Returns:
        The masked mean loss for this block.
    """
    per_token = mask.sum(dim=-1, keepdim=True)
    pairs = mask * (per_token >= 2.0)
    count = pairs.sum(dim=-1, keepdim=True).clamp(min=1.0)

    wanted = torch.log(target.clamp(min=_LOG_TARGET_FLOOR))
    predicted = logits.float()
    centred = (predicted - (predicted * pairs).sum(-1, keepdim=True) / count) - (
        wanted - (wanted * pairs).sum(-1, keepdim=True) / count
    )
    # Huber rather than squared error: the floor puts legitimate targets nine
    # nats from the mean, and a squared penalty there would swamp every token
    # where both modalities still see something.
    elementwise = F.huber_loss(
        centred,
        torch.zeros_like(centred),
        reduction="none",
    )
    return (elementwise * pairs).sum() / pairs.sum().clamp(min=1.0)


def gate_loss(
    gate_logits: list[jt.Float[torch.Tensor, "B T L"]],
    target: jt.Float[torch.Tensor, "B T L"],
    mask: jt.Float[torch.Tensor, "B T L"],
    objective: str = "logit",
) -> jt.Float[torch.Tensor, ""]:
    """Masked supervision of every fusion block\'s gate.

    Args:
        gate_logits: The gate logits of each block, in forward order.
        target: Per-token observability targets.
        mask: Which token/modality pairs carry a measurement.
        objective: ``"logit"`` for the binary cross entropy every result so far
            was trained under, or ``"log"`` for the quantity the
            inverse-variance derivation prescribes.

    Returns:
        The mean loss over the blocks; zero when nothing is supervised.

    Raises:
        ValueError: If the objective is not one this function implements.
    """
    if objective not in ("logit", "log"):
        raise ValueError(
            f"Unknown gate objective {objective!r}; expected 'logit' or 'log'.",
        )
    supervised = mask.sum().clamp(min=1.0)
    total = torch.zeros((), device=target.device, dtype=torch.float32)
    for logits in gate_logits:
        if objective == "log":
            total = total + _centred_log_gate_loss(logits, target, mask)
            continue
        elementwise = F.binary_cross_entropy_with_logits(
            logits.float(),
            target,
            reduction="none",
        )
        total = total + (elementwise * mask).sum() / supervised
    return total / max(len(gate_logits), 1)'''

POLICY_ANCHOR = """            loss["loss_observability_gate"] = gate_loss(
                predictions.observability_gate,
                token_target,
                token_mask,
            )"""

POLICY_BLOCK = """            loss["loss_observability_gate"] = gate_loss(
                predictions.observability_gate,
                token_target,
                token_mask,
                self.lead_config.policy.transfuser.observability_gate_target,
            )"""


def patch(path: pathlib.Path, anchor: str, replacement: str) -> None:
    """Replace one unique anchor, refusing anything ambiguous."""
    text = io.open(path, encoding="utf-8").read()
    found = text.count(anchor)
    if found != 1:
        sys.exit(f"FATAL: {path.name}: anchor appears {found} times, expected 1")
    io.open(path, "w", encoding="utf-8", newline="\n").write(
        text.replace(anchor, replacement))
    print(f"  patched {path.relative_to(ROOT)}")


patch(CONFIG, CONFIG_ANCHOR, CONFIG_BLOCK)
patch(GATE, GATE_ANCHOR, GATE_BLOCK)
patch(POLICY, POLICY_ANCHOR, POLICY_BLOCK)
print("done")
