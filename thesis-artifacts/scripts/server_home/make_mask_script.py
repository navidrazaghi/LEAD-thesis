# -*- coding: utf-8 -*-
"""Build the mask run's driver from rung 2b's, which it replaces.

rung 2b was going to separate the head from the gate inside a null result. This
run spends the same GPU hours asking whether a different intervention on the
same head produces something. Both cannot fit before the machine expires and
the second is the one that can answer the proposal's own question, so the
second is the one that runs.

Derived from rung2b_recipe rather than written fresh: the waiting, the disk
check, the log pinning, the resume-if-a-checkpoint-exists behaviour and the
function-wrapped body are all things that already work, and retyping them is
how a detail goes missing.

Three changes to the overrides. The gate flag goes back on, because the mask
consumes the head's logits and the guard refuses to build without it -- but the
bias never reaches the operator, so this is the head plus a substitution, not
the head plus the gate. The mask flag goes on. Everything else is byte for byte
rung 2a-d's configuration, which is what makes this a one-flag comparison
against it.
"""

import io
import pathlib
import re
import sys

HOME = pathlib.Path.home()
SOURCE = HOME / "run_rung2b_recipe.sh"
TARGET = HOME / "run_mask_recipe.sh"

HEADER = '''#!/bin/bash
#
# The observability mask: the head's signal spent on substitution, not on a bias.
#
# WHY THIS RUN AND NOT RUNG 2B
#
# rung 2b would have separated the head from the gate inside a result that is
# already null: under the published recipe the gate moved the driving score by
# +2.48, -4.81 and -4.08 against the ungated deformable rung, every one inside
# its own standard error, in the second regime to say so. Knowing which half of
# a null is responsible is worth less than one more chance at a result, and
# only one of them fits before the machine expires.
#
# WHAT IS BEING TESTED
#
# The head is not the problem. Against the expert's labels on logs it never
# trained on it reaches 0.160 mean absolute error where a constant predictor
# gets 0.232 for the camera, and 0.193 against 0.369 for LiDAR. The gate takes
# that and biases the operator's modality logits, which reweights: the damaged
# features stay where they are, still travelling the branch trunk into the
# planning decoder that the gate never touches, and when both modalities are
# degraded there is nothing to reweight towards.
#
# This run substitutes instead. Where the head doubts a token's own modality
# the token moves towards a learned per-modality prior, which the fusion can
# read as absent information rather than as a confident reading that is wrong.
#
# WHAT WOULD FALSIFY IT
#
# The prediction is recorded here before the run, so it cannot be fitted
# afterwards: no gain in the intact condition, a moderate one under LiDAR
# damage, and the largest under camera damage and on the weather set, because
# those are where both modalities are unreliable and reweighting is powerless
# by construction. If the intact column improves most, the mechanism is not the
# one claimed and the run says so.
#
# ONE FLAG FROM ITS NEIGHBOUR
#
# Against rung2ad_recipe this is use_observability plus use_observability_mask;
# the gate flag is set only because the mask consumes the head's logits and the
# guard refuses to build without it, and the bias never reaches the operator.
# Everything else -- deformable operator, calibrated references, curriculum, 31
# epochs each stage, batch 32 x accum 2, the same 450 logs -- is identical.
#
# The body sits in a function called on the last line. Bash reads a script by
# byte offset, so editing one while it runs feeds it garbage from the shift.
'''

text = io.open(SOURCE, encoding="utf-8").read()

# --- 1. replace the header, up to the first line of the body ---------------
marker = "\nmain() {"
if marker not in text:
    sys.exit("FATAL: could not find the body marker in the source script")
text = HEADER + text[text.index(marker):]

# --- 2. the overrides ------------------------------------------------------
OLD_FLAGS = """			policy.transfuser.use_observability=true \\
			policy.transfuser.use_observability_gate=false \\"""
NEW_FLAGS = """			policy.transfuser.use_observability=true \\
			policy.transfuser.use_observability_gate=true \\
			policy.transfuser.use_observability_mask=true \\"""
if text.count(OLD_FLAGS) != 2:
    sys.exit(f"FATAL: expected the flag block twice, found {text.count(OLD_FLAGS)}")
text = text.replace(OLD_FLAGS, NEW_FLAGS)

# --- 3. names --------------------------------------------------------------
for old, new in (
    ("rung2b_lead_recipe", "mask_lead_recipe"),
    ("closed_loop_rung2b_recipe.csv", "closed_loop_mask_recipe.csv"),
    ("rung2b_recipe=outputs", "mask_recipe=outputs"),
):
    text = text.replace(old, new)
# Any remaining rung2b-prefixed artefact name -- the per-stage logs, say --
# follows the same rule, so it is renamed by pattern rather than by list.
text = re.sub(r"rung2b_recipe", "mask_recipe", text)

# Nothing may still point at rung 2b's own artefacts.
leftovers = re.findall(r"rung2b\S*", text)
if leftovers:
    sys.exit(f"FATAL: the script still refers to rung 2b: {sorted(set(leftovers))}")

io.open(TARGET, "w", encoding="utf-8", newline="\n").write(text)
TARGET.chmod(0o755)
print(f"wrote {TARGET}")
