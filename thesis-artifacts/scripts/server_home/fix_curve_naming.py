# -*- coding: utf-8 -*-
"""Teach the curve extractor that a post-train directory may carry a suffix.

The stage of a run is read from its directory name, and the rule was that a
name ending in ``_post`` is the second stage. That was true while every
post-train was named that way. ``rung0_lead_recipe_post31`` is not: the 31 in
its name is the epoch budget that distinguishes it from the ten-epoch
post-train beside it, and the rule silently filed a post-train as a pretrain
and gave it a rung of its own.

Nothing failed. The curve was extracted correctly and labelled wrongly, which
is the kind of error that survives into a figure.
"""

import io
import pathlib
import sys

SCRIPT = pathlib.Path.home() / "LEAD/lead/scripts/common/training_curves.py"

ANCHOR = '''        name = directory.name
        stage = "post" if name.endswith("_post") else "pre"
        rung = name[:-5] if stage == "post" else name'''

REPLACEMENT = '''        name = directory.name
        # A post-train may carry a suffix after "_post" -- the epoch budget,
        # for instance, which is what separates rung0_lead_recipe_post31 from
        # the ten-epoch post-train beside it. Matching only the bare "_post"
        # filed that run as a pretrain of a rung of its own.
        suffix = re.search(r"_post\\d*$", name)
        stage = "post" if suffix else "pre"
        rung = name[: suffix.start()] if suffix else name'''

IMPORT_ANCHOR = "import glob\n"
IMPORT_BLOCK = "import glob\nimport re\n"

text = io.open(SCRIPT, encoding="utf-8").read()

if "_post\\d*$" in text:
    sys.exit("already patched; nothing to do")

for anchor in (ANCHOR, IMPORT_ANCHOR):
    if text.count(anchor) != 1:
        sys.exit(f"FATAL: anchor appears {text.count(anchor)} times, expected 1:\n"
                 f"{anchor[:90]}")

text = text.replace(ANCHOR, REPLACEMENT).replace(IMPORT_ANCHOR, IMPORT_BLOCK)
io.open(SCRIPT, "w", encoding="utf-8", newline="\n").write(text)
print("patched training_curves.py")
