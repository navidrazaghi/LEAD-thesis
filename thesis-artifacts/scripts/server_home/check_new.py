"""Build the two new configurations on CPU before any GPU time is spent.

rung1d is rung 2a deformable with the calibrated reference points turned off,
which is the code's own default, so it is one flag from the run that scored
59.74 under LiDAR damage. rung2b adds the observability head to that same run
without the gate that reads it.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LEAD_RUNTIME_TYPE_CHECKING"] = "false"
os.environ["TIMM_USE_OLD_CACHE"] = "1"
os.environ["WANDB_MODE"] = "offline"

import pathlib  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, str(pathlib.Path.home() / "LEAD/lead/src"))

from lead.api.abstract_policy import build_policy  # noqa: E402
from lead.config import load_lead_config  # noqa: E402

DEFORMABLE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion:"
    "DeformableFusionBackbone"
)
BASE = [
    f"policy.transfuser.backbone_target={DEFORMABLE}",
    "training.data.use_sensor_degradation=true",
    "training.optimization.num_epochs=31",
    "training.optimization.batch_size=32",
    "training.lightning.accumulate_grad_batches=2",
]
HEAD_ONLY = [
    "policy.transfuser.deformable_calibrated_reference=true",
    "policy.transfuser.use_observability=true",
    "policy.transfuser.use_observability_gate=false",
]
CASES = {
    "rung1d stage 1 (geometry-free)":
        [*BASE, "policy.transfuser.use_planning_decoder=false"],
    "rung1d stage 2 (geometry-free)":
        [*BASE, "policy.transfuser.use_planning_decoder=true"],
    "rung2b stage 1 (head, no gate)":
        [*BASE, "policy.transfuser.use_planning_decoder=false", *HEAD_ONLY],
    "rung2b stage 2 (head, no gate)":
        [*BASE, "policy.transfuser.use_planning_decoder=true", *HEAD_ONLY],
}

for label, overrides in CASES.items():
    argv = sys.argv
    try:
        sys.argv = [argv[0], *overrides]
        config = load_lead_config(use_cli=True)
    finally:
        sys.argv = argv
    model = build_policy(config)
    transfuser = config.policy.transfuser
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(label)
    print(f"   calibrated={transfuser.deformable_calibrated_reference}"
          f"  obs={transfuser.use_observability}"
          f"  gate={transfuser.use_observability_gate}"
          f"  gates_fusion={getattr(model, 'gates_fusion', None)}")
    print(f"   has observability_decoder {hasattr(model, 'observability_decoder')}")
    print(f"   trainable params {total:,}")
