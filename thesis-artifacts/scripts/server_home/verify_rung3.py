"""Build the rung 3 recipe's model from its exact override strings, on CPU.

The point is to fail here rather than at 3am on the GPU. Every override the
run script passes is fed through the same path train.py uses -- load_lead_config
with use_cli, which raises on an unknown key -- and then build_policy is asked
for the model those overrides describe. A typo, a flag no backbone reads, or a
combination the policy refuses shows up as an exception in this script instead
of as seven hours of nothing.

Nothing here touches the GPU: CUDA_VISIBLE_DEVICES is emptied before torch is
imported, so a stray .cuda() would fail loudly rather than contend with the
training run that is using the card.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LEAD_RUNTIME_TYPE_CHECKING"] = "false"
os.environ["TIMM_USE_OLD_CACHE"] = "1"
os.environ["WANDB_MODE"] = "offline"

import contextlib  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402

ROOT = pathlib.Path.home() / "LEAD/lead"
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from lead.api.abstract_policy import build_policy  # noqa: E402
from lead.config import load_lead_config  # noqa: E402

DEFORMABLE = (
    "lead.policy.transfuser.encoder.backbone_deformable_fusion:"
    "DeformableFusionBackbone"
)

# Exactly the strings run_rung3_recipe.sh passes, minus the paths and the log
# list, which say nothing about whether the model can be built.
RUNG3 = [
    "policy.transfuser.use_planning_decoder=false",
    f"policy.transfuser.backbone_target={DEFORMABLE}",
    "policy.transfuser.deformable_calibrated_reference=true",
    "policy.transfuser.use_observability=true",
    "policy.transfuser.use_observability_gate=true",
    "training.data.use_sensor_degradation=true",
    "training.optimization.num_epochs=31",
    "training.optimization.batch_size=32",
    "training.lightning.accumulate_grad_batches=2",
    "training.data.read_from_cache_store=true",
]

# The same rung with the planning decoder, which is what stage 2 trains and
# what evaluation loads. A model that builds in stage 1 and not in stage 2
# would fail eight hours later, so both are checked.
RUNG3_POST = [
    line.replace("use_planning_decoder=false", "use_planning_decoder=true")
    for line in RUNG3
]

# The deformable rung 2a: the same operator without the head and the gate.
# The parameter difference between this and rung 3 is what the head and the
# gate actually cost.
RUNG2A_DEFORMABLE = [
    line for line in RUNG3_POST
    if "use_observability" not in line
]

# The run that is training right now: dense, curriculum only.
RUNG2A_DENSE = [
    line for line in RUNG2A_DEFORMABLE
    if "backbone_target" not in line and "deformable_calibrated" not in line
]

# Dense with the gate asked for. This must raise: the gate biases a softmax
# over a modality axis and the dense operator has none. Asserting the refusal
# is what makes the claim in the script header a checked one.
DENSE_WITH_GATE = [
    line for line in RUNG3_POST
    if "backbone_target" not in line and "deformable_calibrated" not in line
]


def resolve(overrides: list[str]):
    """Build the config tree exactly as train.py does.

    Args:
        overrides: Dotlist strings, as passed on the command line.

    Returns:
        The resolved config tree.
    """
    argv = sys.argv
    try:
        sys.argv = [argv[0], *overrides]
        return load_lead_config(use_cli=True)
    finally:
        sys.argv = argv


def report(label: str, overrides: list[str]) -> int:
    """Build one model and print what it came out as.

    Args:
        label: Name to print the row under.
        overrides: The overrides defining the run.

    Returns:
        Trainable parameter count, or -1 if the build raised.
    """
    print(f"\n--- {label}")
    try:
        lead_config = resolve(overrides)
    except Exception as error:  # noqa: BLE001
        print(f"  CONFIG REFUSED: {type(error).__name__}: {error}")
        return -1

    transfuser = lead_config.policy.transfuser
    print(f"  backbone            {transfuser.backbone_target.split(':')[-1]}")
    print(f"  calibrated ref      {transfuser.deformable_calibrated_reference}")
    print(f"  observability head  {transfuser.use_observability}")
    print(f"  gate                {transfuser.use_observability_gate}")
    print(f"  planning decoder    {transfuser.use_planning_decoder}")
    print(f"  curriculum          {lead_config.training.data.use_sensor_degradation}")
    print(f"  batch x accum       {lead_config.training.optimization.batch_size}"
          f" x {lead_config.training.lightning.accumulate_grad_batches}")
    print(f"  epochs              {lead_config.training.optimization.num_epochs}")

    try:
        model = build_policy(lead_config)
    except Exception as error:  # noqa: BLE001
        print(f"  BUILD REFUSED: {type(error).__name__}: {error}")
        return -1

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params    {total:,}")
    print(f"  gates fusion        {getattr(model, 'gates_fusion', 'n/a')}")
    for name in ("observability_decoder", "observability_token_targets"):
        print(f"  has {name:<28} {hasattr(model, name)}")
    return total


print("torch", torch.__version__, "| cuda visible:", torch.cuda.is_available())

rung3_pre = report("rung3_recipe, stage 1 (pretrain)", RUNG3)
rung3_post = report("rung3_recipe, stage 2 (post-train)", RUNG3_POST)
rung2ad = report("rung2a deformable recipe (no head, no gate)", RUNG2A_DEFORMABLE)
rung2a_dense = report("rung2a_recipe as running now (dense)", RUNG2A_DENSE)

print("\n--- dense + gate: this MUST be refused")
refused = report("dense with the gate asked for", DENSE_WITH_GATE)
print("  -> refused as expected" if refused == -1 else "  -> NOT REFUSED: the header's claim is wrong")

print("\n=== what the head and the gate cost, in parameters ===")
if rung3_post > 0 and rung2ad > 0:
    print(f"  rung2a deformable   {rung2ad:,}")
    print(f"  rung3               {rung3_post:,}")
    print(f"  head + gate         {rung3_post - rung2ad:,}"
          f"  ({100 * (rung3_post - rung2ad) / rung2ad:.2f}% more)")
if rung2a_dense > 0 and rung2ad > 0:
    print(f"  dense (running now) {rung2a_dense:,}")
    print(f"  deformable - dense  {rung2ad - rung2a_dense:,}"
          f"  ({100 * (rung2ad - rung2a_dense) / rung2a_dense:+.2f}%)")

with contextlib.suppress(Exception):
    print("\ndone")
