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

``calibration``
    Clear-weather routes the scored sets do not use. The caution governor
    adapts a scalar online against an observed risk rate, and both that scalar's
    starting point and the step size that moves it have to come from somewhere.
    If they came from the routes the governor is then scored on, the mechanism
    would be fitted to its own test and any improvement would be unreadable.
    This set is where they come from instead, and it is built by exclusion so
    the disjointness is enforced rather than remembered.

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


def read_route_names(paths: list[pathlib.Path]) -> set[str]:
    """Collect the route file names listed in existing set files.

    Args:
        paths: Route-list files, one file name per line. A path that does not
            exist is skipped, so a first run needs no bootstrapping.

    Returns:
        Every name listed across them.
    """
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"  note: {path} does not exist yet, nothing to exclude from it")
            continue
        names.update(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return names


def report(name: str, routes: list[Route], pool: int, pool_towns: int = 0) -> None:
    """Print what a chosen set covers.

    Args:
        name: The set's name.
        routes: The chosen routes.
        pool: How many were available to choose from.
        pool_towns: How many towns the pool this was drawn from spans; zero
            skips the coverage note.
    """
    towns = collections.Counter(route.town for route in routes)
    print(f"\n{name}: {len(routes)} of {pool} available")
    print("  towns: " + ", ".join(f"{t}={c}" for t, c in towns.most_common()))
    adverse = sum(1 for route in routes if route.is_adverse)
    print(f"  adverse-weather routes in the set: {adverse}")
    if pool_towns and len(towns) < pool_towns:
        # Said out loud because it is a property of the data, not a setting:
        # the clear pool is dominated by one town, so once the scored set has
        # taken the small towns there is nothing left for a disjoint set to
        # spread over. Any conclusion drawn from this set inherits that.
        print(
            f"  note: spans {len(towns)} of the {pool_towns} towns its pool "
            f"offers; the remainder were already taken by the scored sets",
        )


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
        "--calibration-count",
        type=int,
        default=20,
        help="Clear-weather routes to calibrate the caution governor on.",
    )
    parser.add_argument(
        "--exclude",
        type=pathlib.Path,
        nargs="*",
        default=None,
        help=(
            "Route lists the calibration set must not reuse. Defaults to the "
            "scored degradation sets, which is what keeps calibration honest."
        ),
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=here / "src/lead/routes/eval_sets",
        help="Where to write the route lists.",
    )
    arguments = parser.parse_args()
    if arguments.exclude is None:
        arguments.exclude = [
            arguments.out / "degradation.txt",
            arguments.out / "degradation_30.txt",
        ]
    return arguments


def main() -> None:
    """Choose every set and write them out.

    Raises:
        SystemExit: If no routes were found, or the calibration pool cannot
            supply a disjoint set of the requested size.
    """
    args = parse_args()
    routes = read_routes(args.routes)
    if not routes:
        raise SystemExit(f"no routes found under {args.routes}")

    clear = [route for route in routes if not route.is_adverse]
    adverse = [route for route in routes if route.is_adverse]
    print(f"pool: {len(routes)} routes, {len(clear)} clear, {len(adverse)} adverse")

    # The scored degradation set is chosen first and folded into the exclusion,
    # rather than read back from the file it is about to overwrite. Reading the
    # file would compare the calibration set against the *previous* run's
    # choices, so changing --degradation-count would silently let the two
    # overlap.
    degradation = stratify(clear, args.degradation_count)

    print("\nexcluding the scored sets from the calibration pool:")
    scored = read_route_names(args.exclude)
    scored.update(route.path.name for route in degradation)
    print(f"  {len(scored)} route(s) already spoken for")
    calibration_pool = [route for route in clear if route.path.name not in scored]
    if len(calibration_pool) < args.calibration_count:
        raise SystemExit(
            f"only {len(calibration_pool)} clear routes are free of the scored "
            f"sets, but {args.calibration_count} were asked for; lower "
            f"--calibration-count rather than overlapping them.",
        )

    def town_count(pool_routes: list[Route]) -> int:
        return len({route.town for route in pool_routes})

    sets = {
        "degradation": (degradation, len(clear), town_count(clear)),
        "weather": (
            stratify(adverse, args.weather_count),
            len(adverse),
            town_count(adverse),
        ),
        "calibration": (
            stratify(calibration_pool, args.calibration_count),
            len(calibration_pool),
            town_count(clear),
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (chosen, pool, pool_towns) in sets.items():
        report(name, chosen, pool, pool_towns)
        target = args.out / f"{name}.txt"
        target.write_text(
            "\n".join(str(route.path.name) for route in chosen) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {target}")

    # Checked rather than trusted: the exclusion above is the only thing
    # standing between a calibrated mechanism and one fitted to its own test.
    calibrated = {route.path.name for route in sets["calibration"][0]}  # noqa: PD011
    overlap = calibrated & scored
    if overlap:
        raise SystemExit(
            f"calibration set overlaps the scored sets on {sorted(overlap)}; "
            f"this is a bug in the selector, not a configuration problem.",
        )
    print(f"\nverified: calibration set is disjoint from {len(scored)} scored routes")


if __name__ == "__main__":
    main()
