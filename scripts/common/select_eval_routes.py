"""Pick the fixed route sets the closed-loop comparison runs on.

Every model and every condition must be scored on the *same* routes, or the
numbers compare route difficulty rather than models. So the sets are chosen
once, written to disk, and committed.

Two sets, because two different questions are being asked and mixing them
confounds both:

``degradation``
    Clear-weather routes only. The synthetic sensor damage is the single
    variable, so a severity sweep over these traces one clean curve. Rain in
    the simulator on top of a degraded camera would leave you unable to say
    which one moved the score.

``weather``
    The rain, fog and night routes, scored at zero synthetic damage. Training
    saw almost none of these — 0.6% rain, 0.2% night in the released logs
    against roughly half of Bench2Drive — so this is a genuine
    out-of-distribution test that needs no synthetic damage at all.

Selection is stratified by town and deterministic, so re-running reproduces the
same sets.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
from xml.etree import ElementTree

# Same thresholds the training-subset selector uses, so "adverse" means one
# thing across the project.
_RAIN_THRESHOLD = 0.0
_FOG_THRESHOLD = 20.0
_NIGHT_SUN_ALTITUDE = 0.0


class Route:
    """One benchmark route file and the conditions it runs under."""

    def __init__(self, path: pathlib.Path, town: str, weather: dict):
        """Hold a route's identity and worst-case weather.

        Args:
            path: The route XML.
            town: The CARLA map it runs on.
            weather: Worst rain, fog and lowest sun over its keyframes.
        """
        self.path = path
        self.town = town
        self.weather = weather

    @property
    def is_adverse(self) -> bool:
        """Whether the simulator itself makes this route hard to see in."""
        return (
            self.weather["rain"] > _RAIN_THRESHOLD
            or self.weather["fog"] > _FOG_THRESHOLD
            or self.weather["sun"] < _NIGHT_SUN_ALTITUDE
        )


def read_routes(root: pathlib.Path) -> list[Route]:
    """Parse every benchmark route under a directory.

    Args:
        root: Directory of route XMLs.

    Returns:
        The routes, sorted by file name so the order is reproducible.
    """
    routes = []
    for path in sorted(root.glob("*.xml")):
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


def stratify(routes: list[Route], count: int) -> list[Route]:
    """Take ``count`` routes, spread over towns as evenly as the pool allows.

    Round-robin over towns rather than proportional sampling: with a small
    budget, proportional sampling hands almost everything to the two towns
    Bench2Drive is dominated by and leaves the rest unrepresented.

    Args:
        routes: The pool to choose from.
        count: How many to take.

    Returns:
        The chosen routes, sorted by file name.
    """
    by_town: dict[str, list[Route]] = collections.defaultdict(list)
    for route in routes:
        by_town[route.town].append(route)

    chosen: list[Route] = []
    while len(chosen) < count and any(by_town.values()):
        for town in sorted(by_town):
            if not by_town[town]:
                continue
            chosen.append(by_town[town].pop(0))
            if len(chosen) >= count:
                break
    return sorted(chosen, key=lambda route: route.path.name)


def report(name: str, routes: list[Route], pool: int) -> None:
    """Print what a chosen set covers.

    Args:
        name: The set's name.
        routes: The chosen routes.
        pool: How many were available to choose from.
    """
    towns = collections.Counter(route.town for route in routes)
    print(f"\n{name}: {len(routes)} of {pool} available")
    print("  towns: " + ", ".join(f"{t}={c}" for t, c in towns.most_common()))
    adverse = sum(1 for route in routes if route.is_adverse)
    print(f"  adverse-weather routes in the set: {adverse}")


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
        default=here / "src/lead/routes/benchmark_routes/bench2drive",
        help="Benchmark routes to choose from.",
    )
    parser.add_argument(
        "--degradation-count",
        type=int,
        default=40,
        help="Clear-weather routes for the synthetic degradation sweep.",
    )
    parser.add_argument(
        "--weather-count",
        type=int,
        default=40,
        help="Adverse-weather routes for the out-of-distribution test.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=here / "src/lead/routes/eval_sets",
        help="Where to write the route lists.",
    )
    return parser.parse_args()


def main() -> None:
    """Choose both sets and write them out."""
    args = parse_args()
    routes = read_routes(args.routes)
    if not routes:
        raise SystemExit(f"no routes found under {args.routes}")

    clear = [route for route in routes if not route.is_adverse]
    adverse = [route for route in routes if route.is_adverse]
    print(f"pool: {len(routes)} routes, {len(clear)} clear, {len(adverse)} adverse")

    sets = {
        "degradation": (stratify(clear, args.degradation_count), len(clear)),
        "weather": (stratify(adverse, args.weather_count), len(adverse)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (chosen, pool) in sets.items():
        report(name, chosen, pool)
        target = args.out / f"{name}.txt"
        target.write_text(
            "\n".join(str(route.path.name) for route in chosen) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {target}")


if __name__ == "__main__":
    main()
