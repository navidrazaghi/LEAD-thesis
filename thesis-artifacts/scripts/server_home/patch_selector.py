# -*- coding: utf-8 -*-
"""Make the town a rotation axis in the subset selector, not a sort key.

The round-robin below cycles over scenario types, which is why the 450-log
subset carries all fourteen of them with no type below twenty-seven. The town
was handled differently: it was folded into the sort key of each scenario's
queue, so popping from the front exhausted the highest-ranked town before
reaching the second. Two towns took 85% of the budget and six of the twelve
towns the release offers contributed nothing -- including five the scored route
set evaluates on, so eleven of thirty evaluation routes run on maps the model
never saw a frame of.

The intent behind the rank was sound: prefer the towns the benchmark actually
tests. It survives here as the tiebreak. What changes is that a log's position
within its own town leads the key, so the queue reads as every town's first
log, then every town's second, and the round-robin cycles towns the way it
already cycled scenarios.

This is committed unrun. Re-selecting would mean refetching the dataset and
retraining every rung, because the ladder's comparisons are fair only while
every rung shares one training subset.
"""

import io
import pathlib
import sys

SCRIPT = pathlib.Path.home() / "LEAD/lead/scripts/common/fetch_dataset_subset.py"

ANCHOR = '''    queues = {
        scenario: sorted(
            (log for log in group if log.path not in chosen),
            key=lambda log: (rank.get(log.town, len(rank)), log.path),
        )
        for scenario, group in per_scenario.items()
    }'''

REPLACEMENT = '''    ordered = {
        scenario: sorted(
            by_town_position(group, chosen),
            key=lambda pair: (pair[0], rank.get(pair[1].town, len(rank)), pair[1].path),
        )
        for scenario, group in per_scenario.items()
    }
    queues = {
        scenario: [log for _, log in pairs] for scenario, pairs in ordered.items()
    }'''

HELPER = '''def by_town_position(
    group: list[Log],
    chosen: dict[str, Log],
) -> list[tuple[int, Log]]:
    """Number each unchosen log by its position within its own town.

    The index is what turns the town into a rotation axis: sorting on it first
    puts every town's first log ahead of any town's second, so a round-robin
    that pops from the front visits the towns in turn instead of draining the
    highest-ranked one.

    Args:
        group: One scenario type's logs.
        chosen: Logs already taken by an earlier tier, keyed by path.

    Returns:
        Pairs of within-town position and log, for the logs not yet chosen.
    """
    seen: dict[str, int] = {}
    numbered = []
    for log in sorted(group, key=lambda log: log.path):
        if log.path in chosen:
            continue
        position = seen.get(log.town, 0)
        seen[log.town] = position + 1
        numbered.append((position, log))
    return numbered


def select('''

text = io.open(SCRIPT, encoding="utf-8").read()

if "def by_town_position(" in text:
    sys.exit("already patched; nothing to do")

for anchor in (ANCHOR, "def select("):
    if text.count(anchor) != 1:
        sys.exit(f"FATAL: anchor appears {text.count(anchor)} times, expected 1:\n"
                 f"{anchor[:90]}")

text = text.replace(ANCHOR, REPLACEMENT).replace("def select(", HELPER)
io.open(SCRIPT, "w", encoding="utf-8", newline="\n").write(text)
print("patched fetch_dataset_subset.py")
