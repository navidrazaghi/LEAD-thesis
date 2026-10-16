# -*- coding: utf-8 -*-
"""Wire the observability mask into the deformable block.

Four edits, and the third is the one that carries the claim.

The flag joins the observability config beside the gate and the gain it is an
alternative to. The block builds the module when the flag is set. The forward
pass then routes the head's signal to the substitution instead of to the
attention bias -- and passes None where the bias used to go, because a run that
did both would answer neither question.

The gate head is still built and still supervised in mask mode. That is
deliberate: the head, its targets and its loss are what the thesis validated,
so the run differs from a gated run in exactly one thing, what the signal is
used for. The guard in the last edit refuses to build a masked model without
it, because a mask reading logits nobody produces would train something
identical to the control and blame the idea.
"""

import io
import pathlib
import sys

ROOT = pathlib.Path.home() / "LEAD/lead"
BACKBONE = ROOT / "src/lead/policy/transfuser/encoder/backbone_deformable_fusion.py"
CONFIG = ROOT / "src/lead/config/policy/transfuser/observability_config.py"

EDITS: list[tuple[pathlib.Path, str, str]] = []

# --- 1. the flag ----------------------------------------------------------
EDITS.append((
    CONFIG,
    "    use_residual_gain: bool = False\n",
    "    use_residual_gain: bool = False\n"
    "    # Use the gate's own signal to replace unreliable tokens with a\n"
    "    # learned per-modality prior, instead of biasing the modality logits\n"
    "    # with it. Same head, same targets, same loss; only the intervention\n"
    "    # differs, so a masked run is one change against a gated one. Needs\n"
    "    # use_observability_gate for the head that produces the signal, and a\n"
    "    # deformable backbone for the same reason the gate does.\n"
    "    use_observability_mask: bool = False\n",
))

# --- 2. the import --------------------------------------------------------
EDITS.append((
    BACKBONE,
    "from lead.policy.transfuser.encoder.observability_gate import ObservabilityGate\n",
    "from lead.policy.transfuser.encoder.observability_gate import ObservabilityGate\n"
    "from lead.policy.transfuser.encoder.observability_mask import ObservabilityMask\n",
))

# --- 3. the block: build it, and route the signal -------------------------
EDITS.append((
    BACKBONE,
    """        gated: bool,
        gained: bool,
        lead_config: LeadConfig,
    ) -> None:""",
    """        gated: bool,
        gained: bool,
        masked: bool,
        lead_config: LeadConfig,
    ) -> None:""",
))

EDITS.append((
    BACKBONE,
    """            gained: Whether a residual gain scales how much of the attention
                output enters the token.
            lead_config: Root config tree, forwarded to the gate.
        \"\"\"""",
    """            gained: Whether a residual gain scales how much of the attention
                output enters the token.
            masked: Whether the gate's signal replaces unreliable tokens with a
                learned prior instead of biasing the modality logits.
            lead_config: Root config tree, forwarded to the gate.

        Raises:
            ValueError: If ``masked`` is set without ``gated``. The mask reads
                the gate head's logits, so without the head there is nothing to
                read and the block would silently train the control.
        \"\"\"
        if masked and not gated:
            raise ValueError(
                "use_observability_mask needs use_observability_gate: the mask "
                "consumes the gate head's logits, and without that head it "
                "would train a model identical to the ungated control.",
            )""",
))

EDITS.append((
    BACKBONE,
    "        self.residual_gain = ResidualGain(n_embd) if gained else None\n",
    "        self.residual_gain = ResidualGain(n_embd) if gained else None\n"
    "        self.observability_mask = (\n"
    "            ObservabilityMask(n_embd, spatial_shapes) if masked else None\n"
    "        )\n",
))

EDITS.append((
    BACKBONE,
    """        normalized = self.ln1(x)
        gate_logits = self.gate(normalized) if self.gate is not None else None
        attended = self.attn(normalized, gate_logits)""",
    """        normalized = self.ln1(x)
        gate_logits = self.gate(normalized) if self.gate is not None else None
        # Two ways to spend the same signal, and never both: the bias moves
        # where a query reads, the mask changes what is there to be read. A
        # block doing both would leave neither attributable.
        attention_bias = gate_logits
        if self.observability_mask is not None:
            assert gate_logits is not None
            normalized = self.observability_mask(normalized, gate_logits)
            attention_bias = None
        attended = self.attn(normalized, attention_bias)""",
))

# --- 4. the construction site --------------------------------------------
EDITS.append((
    BACKBONE,
    """                    config.use_observability_gate,
                    config.use_residual_gain,
                    lead_config,""",
    """                    config.use_observability_gate,
                    config.use_residual_gain,
                    config.use_observability_mask,
                    lead_config,""",
))


def main() -> None:
    """Apply every edit, refusing anything ambiguous."""
    if "use_observability_mask" in io.open(CONFIG, encoding="utf-8").read():
        sys.exit("already wired; nothing to do")
    for path, anchor, replacement in EDITS:
        text = io.open(path, encoding="utf-8").read()
        found = text.count(anchor)
        if found != 1:
            sys.exit(
                f"FATAL: {path.name}: anchor appears {found} times, "
                f"expected 1:\n{anchor[:160]}",
            )
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            text.replace(anchor, replacement),
        )
        print(f"  patched {path.name}")


main()
