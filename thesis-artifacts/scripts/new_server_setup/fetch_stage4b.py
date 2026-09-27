"""Fetch the 585-log diverse subset, normal view only, at the pinned revision.

Two things make this its own script rather than a plain run of
``scripts/common/fetch_dataset_subset.py``.

The dataset has moved. ``DATASET_PROVENANCE.md`` records the revision every
thesis number was measured against, ``36d36c0`` of 2026-08-09; the branch now
points at ``b9c4f62`` of 2026-09-09. The fetcher resolves against ``main``, so
running it as-is would fetch whatever the branch holds today and quietly put the
thesis numbers and the data behind them on different footing. Everything here
addresses the pinned revision instead, for the listing as well as the files.

And the selection is not recomputed. The provenance says the log list is the
authority, not the algorithm that produced it, so the names come from
``selected_frames_town.txt`` and are checked against the pinned listing before
anything is fetched: a name that no longer resolves is reported and stops the
run, because it would mean this subset is not the one the thesis used.

What is fetched is fetch_selected.py's filter, the script that fetched the
campaign data: both views, and for the camera streams the policy's three
cameras in all four kinds -- camera, depth, instance, semantic. The first pass
here followed fetch_dataset_subset.py's default instead, which drops depth and
the perturbated view, and so did not reproduce the campaign: training reads the
depth cameras, and use_sensor_perturbation swapped in a perturbated sample half
the time in every diverse run. Normal view is 19 files a log, perturbated 15,
31.5 GB in all.

Downloads run four at a time, resume, and are verified against the size the
listing reports. Re-running costs only the checks.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = "ln2697/lead-123d"
PIN = "36d36c020a8838c531999105e65d8a84a33c676f"
API = f"https://huggingface.co/api/datasets/{REPO}"
TREE = f"{API}/tree/{PIN}?recursive=true"
ENDPOINT = "https://hf-mirror.com"
VIEWS = ("logs/normal_view", "logs/perturbated_view")
USED_CAMERAS = ("pcam_l0", "pcam_f0", "pcam_r0")
CAMERA_PREFIXES = ("camera.", "camera_depth.", "camera_instance.", "camera_semantic.")


def next_page(header: str) -> str | None:
    """The 'next' URL out of a Link header, if there is one."""
    for part in header.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None


def fetch_listing(cache: pathlib.Path) -> dict[str, int]:
    """Every file at the pinned revision, with its size. Cached on disk.

    The repo's own fetcher keeps paths only; sizes are what let this script
    verify a download instead of trusting that curl exited cleanly.
    """
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    sizes: dict[str, int] = {}
    url, page = TREE, 0
    while url:
        page += 1
        request = urllib.request.Request(url, headers={"User-Agent": "lead-fetch"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            entries = json.loads(response.read())
            link = response.headers.get("Link", "")
        for entry in entries:
            if entry.get("type") == "file":
                sizes[str(entry["path"])] = int(entry.get("size", 0))
        if page % 50 == 0:
            print(f"  read {page} pages, {len(sizes)} files", flush=True)
        url = next_page(link) or ""
    print(f"  read {page} pages, {len(sizes)} files at {PIN[:7]}", flush=True)
    cache.write_text(json.dumps(sizes), encoding="utf-8")
    return sizes


def wanted(sizes: dict[str, int], names: list[str]) -> list[str]:
    """The files to fetch: the named logs, minus the cameras LEAD never reads."""
    prefixes = tuple(f"{view}/{name}/" for view in VIEWS for name in names)
    paths = []
    for path in sizes:
        if not path.startswith("logs/"):
            continue  # repo-root metadata is added by the caller
        if not path.startswith(prefixes):
            continue
        name = path.split("/")[-1]
        if name.startswith(CAMERA_PREFIXES):
            if not any(f".{camera}." in name for camera in USED_CAMERAS):
                continue
        paths.append(path)
    return sorted(paths)


def wait_for_network() -> None:
    """Block until the machine can reach the endpoint again.

    The captive portal here drops for hours at a time. Without this the rounds
    spin through the whole file list while offline, turning a pause into
    thousands of failed attempts.
    """
    waited = 0
    while True:
        probe = subprocess.run(
            ["curl", "-4", "-sS", "-o", "/dev/null", "--max-time", "20", "-I", ENDPOINT],
            check=False, capture_output=True,
        )
        if probe.returncode == 0:
            if waited:
                print(f"  back online after {waited} minutes", flush=True)
            return
        if waited % 15 == 0:
            print(f"  offline, waiting ({waited} minutes so far)", flush=True)
        waited += 1
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path("data/lead/123D"))
    parser.add_argument("--names", type=pathlib.Path,
                        default=pathlib.Path(
                            "thesis-artifacts/provenance/new_subset/selected_frames_town.txt"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    names = [line.strip() for line in args.names.read_text().splitlines() if line.strip()]
    print(f"{len(names)} log names from {args.names.name}")

    print(f"reading the file listing at the pinned revision {PIN[:7]}")
    sizes = fetch_listing(args.root / "repo_listing_pinned.json")

    # The provenance's own check: a name that no longer resolves means this is
    # not the subset the thesis was measured on.
    present = {"/".join(path.split("/")[:4]) for path in sizes if path.startswith(VIEWS)}
    missing = [f"{view}/{name}" for view in VIEWS for name in names
               if f"{view}/{name}" not in present]
    if missing:
        print(f"FATAL: {len(missing)} of the {len(names)} logs are not at this revision:")
        for name in missing[:10]:
            print(f"  {name}")
        sys.exit(1)
    print(f"all {len(names)} logs resolve at the pinned revision, both views")

    paths = wanted(sizes, names)
    paths += [path for path in sizes if not path.startswith("logs/")]
    total = sum(sizes[path] for path in paths)
    print(f"{len(paths)} files, {total / 1e9:.1f} GB")

    for round_number in range(1, args.rounds + 1):
        todo = [path for path in paths
                if not (args.root / path).exists()
                or (args.root / path).stat().st_size != sizes[path]]
        have = total - sum(sizes[path] for path in todo)
        print(f"round {round_number}: {have / 1e9:.2f} of {total / 1e9:.2f} GB, "
              f"{len(todo)} files to go", flush=True)
        if not todo:
            print("complete")
            break
        wait_for_network()
        running: list[subprocess.Popen] = []
        for index, path in enumerate(todo, 1):
            while len(running) >= args.jobs:
                running = [process for process in running if process.poll() is None]
                time.sleep(0.05)
            target = args.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"{ENDPOINT}/datasets/{REPO}/resolve/{PIN}/{path}"
            # -L is not optional: the endpoint answers 308 then 302 before
            # serving the file, and without it curl writes the redirect body,
            # leaves a zero-byte file and still exits 0.
            # --speed-limit/--speed-time abandon a transfer that has gone
            # quiet, which matters more than it sounds: when the portal drops,
            # every curl otherwise sits on --max-time and the round spends
            # half an hour per file learning the link is down.
            running.append(subprocess.Popen(
                ["curl", "-4", "-fsSL", "-C", "-", "--max-time", "1800",
                 "--connect-timeout", "30",
                 "--speed-limit", "2000", "--speed-time", "60",
                 "-o", str(target), url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if index % 500 == 0:
                print(f"  {index}/{len(todo)} started", flush=True)
        for process in running:
            process.wait()
        time.sleep(5)
    else:
        print("FATAL: still incomplete after every round")
        sys.exit(1)

    done = sum(sizes[path] for path in paths if (args.root / path).exists())
    print(f"on disk: {done / 1e9:.2f} GB in {len(paths)} files")


if __name__ == "__main__":
    main()
