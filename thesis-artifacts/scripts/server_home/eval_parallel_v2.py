# -*- coding: utf-8 -*-
"""Run one evaluation matrix as several CARLA instances at once, then merge.

Version 2 of eval_parallel.py. Two things change, and nothing about how any
single route is driven:

  * Routes are split by expected duration, not dealt round-robin. Route times
    on the 30-route set run from 2.5 to 17 minutes, and round-robin can stack
    the long ones on one shard while the others sit idle waiting for it. On
    the measured times of this machine that tail is most of the gain from a
    fourth shard: 54 min at three shards round-robin, 49 at four round-robin,
    35 at four balanced. Each route goes, longest first, to the shard with the
    least expected work so far (LPT).
  * Four shards by default. At three, the new server ran with GPU compute
    around 20%, 60% of CPU idle, and each shard holding about 10 GB of RAM and
    7.3 GB of GPU memory -- four fit in 62 GB and 40 GB. Five would too, just
    barely, and a shard that runs out of memory costs more than it saves.

Expected durations come from every results CSV on disk that records per-route
wall time: the median for (route, modality) when there is one, else for the
route, else the overall median. Only relative times matter for the split, so
records from the old server serve as well as new ones. Runs that failed for the
infrastructure's reasons are left out, because their times say nothing about
the route.

CARLA runs in synchronous mode, so a route's score does not depend on how fast
the machine is, and which shard drives a route does not change the run. The
three safeguards of version 1 are unchanged: a work directory per shard, ports
four hundred apart, and a merge that keeps one row per
(model, modality, severity, route).

Usage:
  python eval_parallel_v2.py --models NAME=DIR [...] --routes FILE
      --conditions MOD:SEV [...] --out CSV [--shards 4] [--base-port 4000]
      [--plan-only]
"""
import argparse
import csv
import heapq
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "LEAD/lead"
PY = pathlib.Path.home() / "miniconda3/envs/lead/bin/python"
KEY = ("model", "modality", "severity", "route")
# Statuses whose wall time reflects the machine, not the route.
_NOT_TIMINGS = ("NoResult", "Agent timed out", "Agent couldn't be set up", "Agent crashed")
_TIMING_SOURCES = ("thesis-artifacts/results", "results")


def parse_args() -> argparse.Namespace:
    """Command line, mirroring run_evaluation.py's own flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--routes", type=pathlib.Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=4000)
    parser.add_argument("--stagger-seconds", type=int, default=60)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the split and its expected finish time, then stop.",
    )
    return parser.parse_args()


def read_rows(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    """Header and rows of a results CSV, or nothing if it does not exist."""
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def route_timings() -> dict[tuple[str, str], float]:
    """Median wall seconds per (route, modality), and per (route, '*')."""
    samples: dict[tuple[str, str], list[float]] = {}
    for source in _TIMING_SOURCES:
        for path in sorted((ROOT / source).glob("*.csv")):
            header, rows = read_rows(path)
            if "seconds" not in header or "route" not in header:
                continue
            for row in rows:
                if any(marker in (row.get("status") or "") for marker in _NOT_TIMINGS):
                    continue
                try:
                    seconds = float(row["seconds"])
                except (TypeError, ValueError):
                    continue
                if seconds <= 0:
                    continue
                route = row["route"]
                samples.setdefault((route, row.get("modality") or ""), []).append(seconds)
                samples.setdefault((route, "*"), []).append(seconds)
    return {key: statistics.median(values) for key, values in samples.items()}


def expected_seconds(route: str, modalities: list[str], timings: dict) -> float:
    """What one route is expected to cost across every requested condition."""
    overall = statistics.median(timings.values()) if timings else 300.0
    total = 0.0
    for modality in modalities:
        total += timings.get((route, modality), timings.get((route, "*"), overall))
    return total


def balanced_split(routes: list[str], cost: dict[str, float], shards: int) -> list[list[str]]:
    """Longest route first, each to the shard with the least expected work."""
    heap = [(0.0, index) for index in range(shards)]
    split: list[list[str]] = [[] for _ in range(shards)]
    for route in sorted(routes, key=lambda r: (-cost[r], r)):
        load, index = heapq.heappop(heap)
        split[index].append(route)
        heapq.heappush(heap, (load + cost[route], index))
    return split


def main() -> int:
    """Split, launch, wait, merge."""
    args = parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    routes = [line.strip() for line in args.routes.read_text().splitlines() if line.strip()]
    shards = max(1, min(args.shards, len(routes)))
    shard_dir = ROOT / "outputs" / "eval_shards" / out.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    modalities = [pair.split(":", 1)[0] for pair in args.conditions]
    timings = route_timings()
    known = sum(1 for r in routes if (r, "*") in timings)
    cost = {r: expected_seconds(r, modalities, timings) * len(args.models) for r in routes}
    split = balanced_split(routes, cost, shards)
    loads = [sum(cost[r] for r in part) for part in split]
    print(f"timings known for {known} of {len(routes)} routes; "
          f"expected finish {max(loads) / 60:.0f} min "
          f"(shard loads {', '.join(f'{load / 60:.0f}' for load in loads)} min)", flush=True)
    if args.plan_only:
        for index, part in enumerate(split):
            print(f"shard {index}: {len(part)} routes, {loads[index] / 60:.0f} min: {' '.join(part)}")
        return 0

    processes = []
    for index in range(shards):
        route_file = shard_dir / f"routes_{index}.txt"
        route_file.write_text("\n".join(split[index]) + "\n")
        port = args.base_port + 400 * index
        command = [
            str(PY), "scripts/common/run_evaluation.py",
            "--models", *args.models,
            "--routes", str(route_file),
            "--conditions", *args.conditions,
            "--out", str(shard_dir / f"shard_{index}.csv"),
            "--port", str(port),
            "--work-dir", str(ROOT / "outputs" / f"eval_scratch_{out.stem}_{index}"),
        ]
        log = (shard_dir / f"shard_{index}.log").open("a")
        print(f"shard {index}: {len(split[index])} routes on port {port}", flush=True)
        processes.append((index, subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
        # CARLA's start-up spike in memory is the worst moment; keep them apart.
        if index + 1 < shards:
            time.sleep(args.stagger_seconds)

    failed = []
    for index, process in processes:
        if process.wait() != 0:
            failed.append(index)
    print(f"shards finished, failed: {failed or 'none'}", flush=True)

    header, merged = read_rows(out)
    seen = {tuple(row[k] for k in KEY) for row in merged}
    for index in range(shards):
        shard_header, rows = read_rows(shard_dir / f"shard_{index}.csv")
        header = header or shard_header
        for row in rows:
            key = tuple(row[k] for k in KEY)
            if key not in seen:
                seen.add(key)
                merged.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    expected = len(args.models) * len(args.conditions) * len(routes)
    print(f"merged {len(merged)} rows into {out} (matrix is {expected})", flush=True)
    return 1 if failed or len(merged) < expected else 0


if __name__ == "__main__":
    sys.exit(main())
