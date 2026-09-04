"""Choose a training subset of the route set that fits a disk budget.

The full Py123D dataset does not fit on every machine, and a random slice of it
is the wrong thing to keep: it thins the rare scenarios and the rare weather
first, which are exactly what a robustness claim rests on. This picks a subset
by what the routes are for instead.

Three tiers, in order:

1. **Every adverse-weather route.** Rain, fog and night are a few percent of the
   training routes and roughly half of Bench2Drive, so they are the scarce
   resource. None are dropped.
2. **Every route of a rare scenario type.** A type with a handful of routes
   disappears under proportional sampling; the safety-critical benchmark is
   built out of exactly those.
3. **The rest, round-robin over scenario types**, taking benchmark towns first
   within each type. Round-robin keeps every scenario represented as the budget
   shrinks, rather than letting the two largest types eat it.

The result is written as a route-name list. Feed it to
``training.data.py123d_log_names`` to train on, and use it to decide what to
fetch when the dataset itself has to be pulled selectively.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
from xml.etree import ElementTree

# A route counts as adverse if any of its weather keyframes crosses one of
# these. The thresholds are the point where a camera visibly suffers, not the
# point where a human would call it bad weather.
_RAIN_THRESHOLD = 0.0
_FOG_THRESHOLD = 20.0
_NIGHT_SUN_ALTITUDE = 0.0

# A scenario type with fewer routes than this is kept whole.
_RARE_SCENARIO_ROUTES = 30

# Mean bytes per recorded route, from the published 1.1 TB over 8930 routes.
_BYTES_PER_ROUTE = 123 * 1024 * 1024


class Route:
    """One route file, with the facts the selection needs."""

    def __init__(self, path: pathlib.Path, scenario: str, town: str, weather: dict):
        """Hold a route's identity and worst-case weather.

        Args:
            path: The route XML.
            scenario: Its scenario-type directory name.
            town: The CARLA map it runs on.
            weather: Worst rain, fog and lowest sun over its keyframes.
        """
        self.path = path
        self.name = path.stem
        self.scenario = scenario
        self.town = town
        self.weather = weather

    @property
    def is_adverse(self) -> bool:
        """Whether the route carries rain, fog or darkness worth keeping."""
        return (
            self.weather["rain"] > _RAIN_THRESHOLD
            or self.weather["fog"] > _FOG_THRESHOLD
            or self.weather["sun"] < _NIGHT_SUN_ALTITUDE
        )


def read_routes(root: pathlib.Path) -> list[Route]:
    """Parse every route under a directory of scenario-type folders.

    Args:
        root: Directory holding one folder per scenario type.

    Returns:
        The routes, unsorted.
    """
    routes = []
    for path in sorted(root.glob("*/*.xml")):
        try:
            element = ElementTree.parse(path).getroot().find("route")
        except ElementTree.ParseError:
            continue
        if element is None or element.get("town") is None:
            continue
        keyframes = element.find("weathers")
        frames = keyframes.findall("weather") if keyframes is not None else []
        routes.append(
            Route(
                path,
                path.parent.name,
                str(element.get("town")),
                {
                    "rain": max(
                        [float(f.get("precipitation", 0)) for f in frames] or [0.0],
                    ),
                    "fog": max(
                        [float(f.get("fog_density", 0)) for f in frames] or [0.0],
                    ),
                    "sun": min(
                        [float(f.get("sun_altitude_angle", 90)) for f in frames]
                        or [90.0],
                    ),
                },
            ),
        )
    return routes


def benchmark_town_order(benchmark_root: pathlib.Path) -> list[str]:
    """Towns ranked by how much of the closed-loop benchmarks runs on them.

    Args:
        benchmark_root: Directory holding the benchmark route folders.

    Returns:
        Town names, most-evaluated first.
    """
    counts: collections.Counter[str] = collections.Counter()
    for path in benchmark_root.glob("*/*.xml"):
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            continue
        for element in root.findall("route"):
            town = element.get("town")
            if town:
                counts[town] += 1
    return [town for town, _ in counts.most_common()]


def select(routes: list[Route], budget: int, town_order: list[str]) -> list[Route]:
    """Pick ``budget`` routes by the three tiers described in the module docstring.

    Args:
        routes: Every candidate route.
        budget: How many routes to end up with.
        town_order: Towns most worth keeping, best first.

    Returns:
        The chosen routes.
    """
    per_scenario: dict[str, list[Route]] = collections.defaultdict(list)
    for route in routes:
        per_scenario[route.scenario].append(route)

    town_rank = {town: index for index, town in enumerate(town_order)}
    chosen: dict[str, Route] = {}

    # Tier 1 and 2: what proportional sampling would destroy.
    for route in routes:
        if route.is_adverse:
            chosen[route.name] = route
    for group in per_scenario.values():
        if len(group) < _RARE_SCENARIO_ROUTES:
            for route in group:
                chosen[route.name] = route
    if len(chosen) >= budget:
        return sorted(chosen.values(), key=lambda r: (r.scenario, r.name))

    # Tier 3: round-robin over scenario types, benchmark towns first.
    queues = {
        scenario: sorted(
            (r for r in group if r.name not in chosen),
            key=lambda r: (town_rank.get(r.town, len(town_rank)), r.name),
        )
        for scenario, group in per_scenario.items()
    }
    while len(chosen) < budget and any(queues.values()):
        for scenario in sorted(queues):
            queue = queues[scenario]
            if not queue:
                continue
            route = queue.pop(0)
            chosen[route.name] = route
            if len(chosen) >= budget:
                break
    return sorted(chosen.values(), key=lambda r: (r.scenario, r.name))


def report(selected: list[Route], everything: list[Route]) -> None:
    """Print what the selection kept, against what it had to choose from.

    Args:
        selected: The chosen routes.
        everything: Every candidate route.
    """
    gigabytes = len(selected) * _BYTES_PER_ROUTE / 1024**3
    print(
        f"selected {len(selected)} of {len(everything)} routes "
        f"({100 * len(selected) / len(everything):.1f}%), "
        f"about {gigabytes:.0f} GB of recorded data",
    )

    def share(rows: list[Route], predicate) -> str:
        kept = sum(1 for r in rows if predicate(r))
        return f"{kept}"

    print("\nadverse weather kept in full:")
    for label, predicate in (
        ("rain", lambda r: r.weather["rain"] > _RAIN_THRESHOLD),
        ("fog", lambda r: r.weather["fog"] > _FOG_THRESHOLD),
        ("night", lambda r: r.weather["sun"] < _NIGHT_SUN_ALTITUDE),
    ):
        print(
            f"  {label:6s} {share(selected, predicate):>5s} of "
            f"{share(everything, predicate):>5s}",
        )

    print("\nscenario coverage:")
    per_scenario_all = collections.Counter(r.scenario for r in everything)
    per_scenario_kept = collections.Counter(r.scenario for r in selected)
    for scenario in sorted(per_scenario_all):
        kept, total = per_scenario_kept[scenario], per_scenario_all[scenario]
        print(f"  {scenario:<42s} {kept:>4d} / {total:<5d}")

    print("\ntowns:")
    per_town_all = collections.Counter(r.town for r in everything)
    per_town_kept = collections.Counter(r.town for r in selected)
    for town, total in per_town_all.most_common():
        print(f"  {town:<12s} {per_town_kept[town]:>4d} / {total:<5d}")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    here = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--routes",
        type=pathlib.Path,
        default=here / "src/lead/routes/data_routes/lead",
        help="Directory of scenario-type folders to choose from.",
    )
    parser.add_argument(
        "--benchmarks",
        type=pathlib.Path,
        default=here / "src/lead/routes/benchmark_routes",
        help="Benchmark routes, used to rank towns by evaluation weight.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=700,
        help="How many routes to keep.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Write the chosen route names here, one per line.",
    )
    return parser.parse_args()


def main() -> None:
    """Select a subset and report what it covers."""
    args = parse_args()
    routes = read_routes(args.routes)
    if not routes:
        raise SystemExit(f"no routes found under {args.routes}")
    selected = select(routes, args.budget, benchmark_town_order(args.benchmarks))
    report(selected, routes)
    if args.out is not None:
        args.out.write_text(
            "\n".join(route.name for route in selected) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {len(selected)} route names to {args.out}")


if __name__ == "__main__":
    main()
