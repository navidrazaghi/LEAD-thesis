# -*- coding: utf-8 -*-
"""Run one evaluation matrix as several CARLA instances at once, then merge.

run_evaluation.py drives one CARLA server through every route in turn, and on
this machine a 90-row matrix takes about seven hours while the GPU sits mostly
idle. The routes are independent, so they can be split.

Three things make that safe, and each is handled here rather than assumed:

  * The scratch directory. Every route clears ``outputs/eval_scratch`` before
    it starts and reads its score back from whatever checkpoint file it finds
    there. Two instances sharing it would delete each other's runs, or read
    each other's score, with nothing to show for it. Each shard gets its own
    ``--work-dir``.
  * Ports. CARLA takes its world port and the next one for streaming, and the
    traffic manager is probed from the world port plus a hundred. Shards are
    spaced four hundred apart so none of those ranges can meet.
  * The results file. Shards append to their own CSVs, and the merge keeps a
    row only once per (model, modality, severity, route), so rerunning after a
    partial failure resumes instead of duplicating.

Usage:
  python eval_parallel.py --models NAME=DIR [...] --routes FILE
      --conditions MOD:SEV [...] --out CSV [--shards 3] [--base-port 4000]
"""
import argparse
import csv
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path.home() / "LEAD/lead"
PY = pathlib.Path.home() / "miniconda3/envs/lead/bin/python"
KEY = ("model", "modality", "severity", "route")


def parse_args() -> argparse.Namespace:
    """Command line, mirroring run_evaluation.py's own flags."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--routes", type=pathlib.Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--base-port", type=int, default=4000)
    parser.add_argument("--stagger-seconds", type=int, default=60)
    return parser.parse_args()


def read_rows(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    """Header and rows of a results CSV, or nothing if it does not exist."""
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    """Split, launch, wait, merge."""
    args = parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    routes = [line.strip() for line in args.routes.read_text().splitlines() if line.strip()]
    shards = max(1, min(args.shards, len(routes)))
    shard_dir = ROOT / "outputs" / "eval_shards" / out.stem
    shard_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for index in range(shards):
        route_file = shard_dir / f"routes_{index}.txt"
        route_file.write_text("\n".join(routes[index::shards]) + "\n")
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
        print(f"shard {index}: {len(routes[index::shards])} routes on port {port}", flush=True)
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
