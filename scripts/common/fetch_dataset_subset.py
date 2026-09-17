"""Fetch a stratified subset of the Py123D dataset over plain HTTPS.

Two problems this works around. The full release does not fit on a machine with
under 100 GB free, and on some networks the ``huggingface_hub`` client dies
mid-transfer (``RemoteProtocolError: Server disconnected``) while a plain
``curl`` on the same URL succeeds. So this resolves the file list through the
dataset API, decides what is worth keeping, and pulls each file with curl,
resumably.

What it keeps is decided the same way as
:mod:`scripts.common.select_training_subset`: every adverse-weather route, every
route of a rare scenario type, then a round-robin fill over scenario types. See
that module for why a random slice is the wrong subset.

On top of that it drops files the model never opens. LEAD reads three of the six
cameras, so the other three are pure download; depth carries a loss weight of
1e-5, so it is optional too. Together that is roughly 40% of each log.

    python scripts/common/fetch_dataset_subset.py --budget 450 --dry-run
    python scripts/common/fetch_dataset_subset.py --budget 450 --jobs 4
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import urllib.request
from xml.etree import ElementTree

REPO = "ln2697/lead-123d"
API = f"https://huggingface.co/api/datasets/{REPO}"
# huggingface.co redirects anything large to us.aws.cdn.hf.co, which some
# networks cannot reach: the transfer then burns a 130 s timeout per file
# before failing. The mirror bounces back to the same origin over a route that
# does resolve, so it is the default here.
DEFAULT_ENDPOINT = "https://hf-mirror.com"

# A log directory is "<Town>_Rep<n>_<route stem>_route0_<timestamp>"; the stem
# is what names the route XML the log was recorded from.
_LOG_NAME = re.compile(r"^(?P<town>Town\w+?)_Rep\d+_(?P<stem>.+)_route0_[\d_]+$")

# Weather thresholds; see select_training_subset for the reasoning.
_RAIN_THRESHOLD = 0.0
_FOG_THRESHOLD = 20.0
_NIGHT_SUN_ALTITUDE = 0.0
_RARE_SCENARIO_LOGS = 30

# Cameras the model actually ingests, matching TransfuserCameraConfig.
_USED_CAMERAS = ("pcam_l0", "pcam_f0", "pcam_r0")

# 38 of the released logs ship without sync.arrow, the table that aligns the
# modalities on a common timestamp. The scene index finds nothing in such a log
# and the cache build dies on it with "no scenes match splits", so they are
# skipped rather than downloaded and tripped over later.
_REQUIRED_FILES = ("sync.arrow", "ego_state_se3.arrow", "lidar.lidar_top.arrow")


def fetch_listing(cache: pathlib.Path) -> list[str]:
    """The repo's file list, downloaded once and cached on disk.

    Args:
        cache: Where to keep the API response.

    Returns:
        Every file path in the repo.
    """
    if not cache.exists():
        with urllib.request.urlopen(API, timeout=120) as response:  # noqa: S310
            cache.write_bytes(response.read())
    payload = json.loads(cache.read_text(encoding="utf-8"))
    return [s["rfilename"] for s in payload.get("siblings", [])]


def route_weather(route_roots: list[pathlib.Path]) -> dict[str, dict]:
    """Worst-case weather of every route XML, keyed by file stem.

    Args:
        route_roots: Directories of route sets to scan recursively.

    Returns:
        Rain, fog and lowest sun angle per route stem.
    """
    weather: dict[str, dict] = {}
    for root in route_roots:
        for path in root.rglob("*.xml"):
            try:
                element = ElementTree.parse(path).getroot().find("route")
            except ElementTree.ParseError:
                continue
            if element is None:
                continue
            frames_parent = element.find("weathers")
            frames = (
                frames_parent.findall("weather") if frames_parent is not None else []
            )
            weather[path.stem] = {
                "rain": max(
                    [float(f.get("precipitation", 0)) for f in frames] or [0.0]
                ),
                "fog": max([float(f.get("fog_density", 0)) for f in frames] or [0.0]),
                "sun": min(
                    [float(f.get("sun_altitude_angle", 90)) for f in frames] or [90.0],
                ),
            }
    return weather


class Log:
    """One recorded log in the release, with what the selection needs."""

    def __init__(self, path: str, scenario: str, town: str, weather: dict):
        """Hold a log's identity and the weather of the route behind it.

        Args:
            path: Repo-relative log directory.
            scenario: Scenario-type folder it sits in.
            town: CARLA map, read off the log name.
            weather: Worst rain, fog and lowest sun; empty when unknown.
        """
        self.path = path
        self.scenario = scenario
        self.town = town
        self.weather = weather

    @property
    def is_adverse(self) -> bool:
        """Whether the log carries rain, fog or darkness worth keeping."""
        if not self.weather:
            return False
        return (
            self.weather["rain"] > _RAIN_THRESHOLD
            or self.weather["fog"] > _FOG_THRESHOLD
            or self.weather["sun"] < _NIGHT_SUN_ALTITUDE
        )


def build_logs(files: list[str], weather: dict[str, dict]) -> list[Log]:
    """Group the file list into logs and attach each one's weather.

    Logs the release published incomplete are dropped here, so a broken one
    never reaches the cache build.

    Args:
        files: Every repo file path.
        weather: Route weather keyed by route stem.

    Returns:
        One entry per usable log directory.
    """
    contents: dict[str, set[str]] = collections.defaultdict(set)
    for path in files:
        parts = path.split("/")
        if path.startswith("logs/") and len(parts) >= 5:
            contents["/".join(parts[:4])].add(parts[4])

    skipped = 0
    logs = []
    for directory in sorted(contents):
        parts = directory.split("/")
        if len(parts) < 4:
            continue
        if not all(name in contents[directory] for name in _REQUIRED_FILES):
            skipped += 1
            continue
        match = _LOG_NAME.match(parts[3])
        if match is None:
            continue
        logs.append(
            Log(
                directory,
                parts[2],
                match.group("town"),
                weather.get(match.group("stem"), {}),
            ),
        )
    if skipped:
        print(f"skipped {skipped} logs the release published incomplete")
    return logs


def select(logs: list[Log], budget: int, town_order: list[str]) -> list[Log]:
    """Choose ``budget`` logs, adverse weather and rare scenarios first.

    Args:
        logs: Every available log.
        budget: How many to keep.
        town_order: Towns most worth keeping, best first.

    Returns:
        The chosen logs.
    """
    per_scenario: dict[str, list[Log]] = collections.defaultdict(list)
    for log in logs:
        per_scenario[log.scenario].append(log)

    rank = {town: index for index, town in enumerate(town_order)}
    chosen: dict[str, Log] = {}
    for log in logs:
        if log.is_adverse:
            chosen[log.path] = log
    for group in per_scenario.values():
        if len(group) < _RARE_SCENARIO_LOGS:
            for log in group:
                chosen[log.path] = log

    queues = {
        scenario: sorted(
            (log for log in group if log.path not in chosen),
            key=lambda log: (rank.get(log.town, len(rank)), log.path),
        )
        for scenario, group in per_scenario.items()
    }
    while len(chosen) < budget and any(queues.values()):
        for scenario in sorted(queues):
            if not queues[scenario]:
                continue
            log = queues[scenario].pop(0)
            chosen[log.path] = log
            if len(chosen) >= budget:
                break
    return sorted(chosen.values(), key=lambda log: log.path)


def wanted_files(files: list[str], selected: list[Log], keep_depth: bool) -> list[str]:
    """The repo files to actually fetch for the chosen logs.

    Args:
        files: Every repo file path.
        selected: The chosen logs.
        keep_depth: Whether to fetch the depth cameras.

    Returns:
        Repo-relative paths, plus the repo-root metadata files.
    """
    keep_prefixes = {log.path for log in selected}
    wanted = [p for p in files if not p.startswith("logs/")]
    for path in files:
        if "/".join(path.split("/")[:4]) not in keep_prefixes:
            continue
        name = path.split("/")[-1]
        if name.startswith(
            ("camera.", "camera_depth.", "camera_instance.", "camera_semantic.")
        ):
            if not any(f".{camera}." in name for camera in _USED_CAMERAS):
                continue
            if not keep_depth and name.startswith("camera_depth."):
                continue
        wanted.append(path)
    return wanted


def download(paths: list[str], root: pathlib.Path, jobs: int, endpoint: str) -> int:
    """Fetch each file with curl, resuming and skipping what is already there.

    Args:
        paths: Repo-relative file paths.
        root: Local dataset root the repo layout is mirrored under.
        jobs: How many curl processes to run at once.
        endpoint: Host to resolve the files through.

    Returns:
        The number of files that failed.
    """
    resolve = f"{endpoint}/datasets/{REPO}/resolve/main"
    pending = [p for p in paths if not (root / p).exists()]
    print(f"{len(paths) - len(pending)} already present, {len(pending)} to fetch")
    failures = 0
    running: list[tuple[subprocess.Popen, str]] = []

    def reap(limit: int) -> None:
        nonlocal failures
        while len(running) >= limit:
            for index, (process, path) in enumerate(running):
                if process.poll() is not None:
                    if process.returncode != 0:
                        failures += 1
                        print(f"  FAILED {path}", file=sys.stderr)
                    running.pop(index)
                    break

    for done, path in enumerate(pending, 1):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        reap(jobs)
        running.append(
            (
                subprocess.Popen(  # noqa: S603
                    [
                        "curl",
                        "-sSL",
                        "--fail",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "3",
                        "--retry-all-errors",
                        "--connect-timeout",
                        "20",
                        "-C",
                        "-",
                        "-o",
                        str(target),
                        f"{resolve}/{path}",
                    ],
                    stdout=subprocess.DEVNULL,
                ),
                path,
            ),
        )
        if done % 50 == 0:
            print(f"  queued {done}/{len(pending)}")
    reap(1)
    return failures


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    here = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=450, help="Logs to keep.")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=here / "data/lead/123D",
        help="Local dataset root; must match PY123D_DATA_ROOT.",
    )
    parser.add_argument(
        "--routes",
        type=pathlib.Path,
        default=here / "src/lead/routes/data_routes",
        help="Route sets to read weather from.",
    )
    parser.add_argument(
        "--benchmarks",
        type=pathlib.Path,
        default=here / "src/lead/routes/benchmark_routes",
        help="Benchmark routes, used to rank towns by evaluation weight.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent transfers.")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Host to fetch files through; the mirror avoids the blocked CDN.",
    )
    parser.add_argument(
        "--keep-depth",
        action="store_true",
        help="Fetch the depth cameras too; their loss weight is 1e-5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selection and stop.",
    )
    return parser.parse_args()


def main() -> None:
    """Select a subset and fetch it."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = fetch_listing(args.out / "repo_listing.json")
    weather = route_weather([args.routes])
    logs = build_logs(files, weather)

    towns: collections.Counter[str] = collections.Counter()
    for path in args.benchmarks.rglob("*.xml"):
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            continue
        for element in root.findall("route"):
            if element.get("town"):
                towns[str(element.get("town"))] += 1

    selected = select(logs, args.budget, [t for t, _ in towns.most_common()])
    paths = wanted_files(files, selected, args.keep_depth)

    print(f"release: {len(logs)} logs, {len(files)} files")
    print(f"selected: {len(selected)} logs, {len(paths)} files")
    known = [log for log in selected if log.weather]
    print(f"  weather known for {len(known)} of them")
    print(f"  adverse kept: {sum(1 for log in selected if log.is_adverse)}")
    print("\nscenarios:")
    for scenario, count in collections.Counter(
        log.scenario for log in selected
    ).most_common():
        print(f"   {count:5d}  {scenario}")
    print("\ntowns:")
    for town, count in collections.Counter(log.town for log in selected).most_common():
        print(f"   {count:5d}  {town}")

    if args.dry_run:
        print("\ndry run; nothing fetched")
        return
    failures = download(paths, args.out, args.jobs, args.endpoint)
    print(f"\ndone, {failures} failures")


if __name__ == "__main__":
    main()
