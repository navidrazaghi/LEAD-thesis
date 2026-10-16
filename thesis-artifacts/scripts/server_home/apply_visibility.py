"""Wire the weather-visibility head into the policy, behind a flag."""

import io
import pathlib
import sys

ROOT = pathlib.Path.home() / "LEAD/lead"
OBS = ROOT / "src/lead/config/policy/transfuser/observability_config.py"
WEIGHTS = ROOT / "src/lead/config/policy/transfuser/transfuser_config.py"
POLICY = ROOT / "src/lead/policy/transfuser/transfuser.py"

OBS_ANCHOR = '''    observability_gate_target: str = "logit"'''

OBS_BLOCK = '''    observability_gate_target: str = "logit"
    # Predict the weather visibility class the expert conditions on. The label
    # is in every batch already and no run has read it; the expert changes its
    # lane-change transition and its target speed under LIMITED and
    # VERY_LIMITED, so the student currently imitates those decisions without
    # seeing what caused them. Off by default: no result here was trained with
    # it.
    use_weather_visibility: bool = False
    # Weight of that head relative to the other tasks. The per-task weights are
    # normalized by their sum, so this quietly shrinks the driving losses like
    # every other auxiliary task does.
    weather_visibility_loss_weight: float = 1.0
    # Width of the head's single hidden layer.
    weather_visibility_head_channels: int = 64'''

WEIGHTS_ANCHOR = '''            "loss_observability_gate": self.observability_gate_loss_weight,'''

WEIGHTS_BLOCK = '''            "loss_observability_gate": self.observability_gate_loss_weight,
            "loss_weather_visibility": self.weather_visibility_loss_weight,'''

OFF_ANCHOR = '''        if not self.use_observability:
            weights["loss_observability"] = 0.0'''

OFF_BLOCK = '''        if not self.use_observability:
            weights["loss_observability"] = 0.0

        if not self.use_weather_visibility:
            weights["loss_weather_visibility"] = 0.0'''

IMPORT_ANCHOR = (
    "from lead.policy.transfuser.decoder.observability_decoder import "
    "ObservabilityDecoder"
)
IMPORT_BLOCK = (
    "from lead.policy.transfuser.decoder.observability_decoder import "
    "ObservabilityDecoder\n"
    "from lead.policy.transfuser.decoder.visibility_decoder import VisibilityDecoder"
)

BUILD_ANCHOR = '''        if self.config.use_observability:
            self.observability_decoder = ObservabilityDecoder(lead_config)'''
BUILD_BLOCK = '''        if self.config.use_observability:
            self.observability_decoder = ObservabilityDecoder(lead_config)

        self.visibility_decoder = None
        if self.config.use_weather_visibility:
            self.visibility_decoder = VisibilityDecoder(lead_config)'''

INIT_ANCHOR = """        pred_observability = None"""
INIT_BLOCK = """        pred_observability = None
        pred_weather_visibility = None"""

FORWARD_ANCHOR = '''            if self.config.use_observability:
                pred_observability = self.observability_decoder(bev_feature_grid)'''
FORWARD_BLOCK = '''            if self.config.use_observability:
                pred_observability = self.observability_decoder(bev_feature_grid)

            if self.visibility_decoder is not None:
                pred_weather_visibility = self.visibility_decoder(bev_feature_grid)'''

PRED_ANCHOR = """            observability=pred_observability,"""
PRED_BLOCK = """            observability=pred_observability,
            weather_visibility=pred_weather_visibility,"""

FIELD_ANCHOR = """    # Per-modality observability logits over the BEV cell grid.
    observability: jt.Float[torch.Tensor, "bs n_modalities cell_h cell_w"] | None"""
FIELD_BLOCK = """    # Per-modality observability logits over the BEV cell grid.
    observability: jt.Float[torch.Tensor, "bs n_modalities cell_h cell_w"] | None
    # Scores over the four weather visibility classes, for the whole frame.
    weather_visibility: jt.Float[torch.Tensor, "bs 4"] | None"""

LOSS_ANCHOR = '''        # Observability loss
        if self.config.use_observability:'''
LOSS_BLOCK = '''        # Weather visibility loss
        if self.visibility_decoder is not None:
            assert predictions.weather_visibility is not None
            self.visibility_decoder.compute_loss(
                predictions.weather_visibility,
                batch,
                loss,
                log=auxiliary_log,
            )

        # Observability loss
        if self.config.use_observability:'''


def patch(path: pathlib.Path, pairs: list[tuple[str, str]]) -> None:
    """Apply anchor/replacement pairs, refusing anything ambiguous."""
    text = io.open(path, encoding="utf-8").read()
    for anchor, replacement in pairs:
        found = text.count(anchor)
        if found != 1:
            sys.exit(
                f"FATAL: {path.name}: anchor appears {found} times, expected 1:\n"
                f"{anchor[:90]}")
        text = text.replace(anchor, replacement)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)
    print(f"  patched {path.relative_to(ROOT)}")


patch(OBS, [(OBS_ANCHOR, OBS_BLOCK)])
patch(WEIGHTS, [(WEIGHTS_ANCHOR, WEIGHTS_BLOCK), (OFF_ANCHOR, OFF_BLOCK)])
patch(POLICY, [
    (IMPORT_ANCHOR, IMPORT_BLOCK),
    (BUILD_ANCHOR, BUILD_BLOCK),
    (INIT_ANCHOR, INIT_BLOCK),
    (FORWARD_ANCHOR, FORWARD_BLOCK),
    (PRED_ANCHOR, PRED_BLOCK),
    (FIELD_ANCHOR, FIELD_BLOCK),
    (LOSS_ANCHOR, LOSS_BLOCK),
])
print("done")
