# -*- coding: utf-8 -*-
"""Give the two curve readers the type hints the repository's linter asks for.

Both failures predate the stage-naming fix -- the file has been failing
pydoclint since it was written -- but the hooks run on commit, so the fix
cannot be committed while they stand. Both functions take the same thing: the
path of one offline run's ``run-*.wandb`` file.
"""

import io
import pathlib
import sys

SCRIPT = pathlib.Path.home() / "LEAD/lead/scripts/common/training_curves.py"

EDITS = [
    (
        "def read_history(path):",
        "def read_history(path: pathlib.Path) -> Iterator[dict]:",
    ),
    (
        "def per_epoch(path):",
        "def per_epoch(path: pathlib.Path) -> dict[int, tuple[float, float, int]]:",
    ),
    (
        "import csv\nimport glob\n",
        "import csv\nimport glob\n",
    ),
]

IMPORT_ANCHOR = "import csv\n"
IMPORT_BLOCK = "import csv\nfrom collections.abc import Iterator\n"

text = io.open(SCRIPT, encoding="utf-8").read()

if "def read_history(path: pathlib.Path)" in text:
    sys.exit("already hinted; nothing to do")

for anchor, replacement in EDITS:
    if text.count(anchor) != 1:
        sys.exit(f"FATAL: anchor appears {text.count(anchor)} times, expected 1:\n"
                 f"{anchor[:80]}")
    text = text.replace(anchor, replacement)

if text.count(IMPORT_ANCHOR) != 1:
    sys.exit(f"FATAL: import anchor appears {text.count(IMPORT_ANCHOR)} times")
text = text.replace(IMPORT_ANCHOR, IMPORT_BLOCK)

io.open(SCRIPT, "w", encoding="utf-8", newline="\n").write(text)
print("added type hints to read_history and per_epoch")
