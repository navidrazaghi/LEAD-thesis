"""Add the checkpoints/transfuser note to the reference provenance file."""

import io
import pathlib
import sys

DOC = pathlib.Path.home() / "LEAD/lead/thesis-artifacts/provenance/reference_checkpoints.md"

ANCHOR = "## What reads them\n"

BLOCK = """## `checkpoints/transfuser` is the same model again

The README drives its four example commands against
`--checkpoint checkpoints/transfuser`, which is a separate 264 MB directory
outside `reference/`. It is not a separate model. Its config records the same
upstream run as `reference/seed0` --
`001_training/posttrain/260807_133603` -- so the two directories hold one
checkpoint, fetched twice: into `checkpoints/transfuser` on 2026-09-15 and into
`reference/seed0` on 2026-09-25.

That matters on a fresh clone. The README's commands do not work until those
264 MB are back, and nothing in the repository fetches them; whoever restores
this project has to obtain the checkpoint and can then satisfy both paths from
one download.

## What reads them
"""

text = io.open(DOC, encoding="utf-8").read()
if "checkpoints/transfuser` is the same model again" in text:
    sys.exit("already present; nothing to do")
if text.count(ANCHOR) != 1:
    sys.exit(f"FATAL: anchor appears {text.count(ANCHOR)} times, expected 1")
io.open(DOC, "w", encoding="utf-8", newline="\n").write(text.replace(ANCHOR, BLOCK))
print(f"inserted into {DOC.name}")
