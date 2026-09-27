"""How many bytes the 585-log subset actually is.

The fetcher's dry run reports how many logs and files it would take but not how
large they are, and the cached repository listing keeps paths only -- the sizes
the API returned with them were dropped. Re-walking all 546 pages to recover
them would cost another twenty minutes, so this measures instead: it asks the
server for the size of every file of a sample of the chosen logs, and scales the
mean log up to all 585.

The sample is what makes the answer trustworthy, so it is reported alongside the
estimate: with the spread between logs printed, it is clear how much the total
could move. Logs differ mainly by route length, which is why whole logs are
sampled rather than individual files.

Run with the repo as the working directory. Nothing is downloaded.
"""
import collections
import importlib.util
import pathlib
import random
import statistics
import subprocess
import sys
from xml.etree import ElementTree

REPO = "ln2697/lead-123d"
ENDPOINT = "https://hf-mirror.com"
SAMPLE = 14
BUDGET = 585


def load_fetcher():
    """The fetcher module, imported from its path."""
    path = pathlib.Path("scripts/common/fetch_dataset_subset.py").resolve()
    spec = importlib.util.spec_from_file_location("fetch_dataset_subset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head_size(path: str) -> int:
    """The Content-Length the server reports for one repo file."""
    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{path}"
    result = subprocess.run(
        ["curl", "-4", "-sSIL", "--max-time", "60", url],
        capture_output=True, text=True, check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return 0


def main() -> None:
    fetcher = load_fetcher()
    here = pathlib.Path(".").resolve()
    files = fetcher.fetch_listing(here / "data/lead/123D/repo_listing.json")
    weather = fetcher.route_weather([here / "src/lead/routes/data_routes"])
    logs = fetcher.build_logs(files, weather)

    # The town ranking main() builds: how often each town appears in the
    # benchmark routes, best first.
    towns = collections.Counter()
    for path in (here / "src/lead/routes/benchmark_routes").rglob("*.xml"):
        try:
            root = ElementTree.parse(path).getroot()
        except ElementTree.ParseError:
            continue
        for element in root.findall("route"):
            if element.get("town"):
                towns[str(element.get("town"))] += 1
    selected = fetcher.select(logs, BUDGET, [t for t, _ in towns.most_common()])
    wanted = fetcher.wanted_files(files, selected, keep_depth=False)
    print(f"selected {len(selected)} logs, {len(wanted)} files")

    by_log: dict[str, list[str]] = {}
    for path in wanted:
        if path.startswith("logs/"):
            by_log.setdefault("/".join(path.split("/")[:4]), []).append(path)
    print(f"{len(by_log)} log directories")

    random.seed(0)
    sample = random.sample(sorted(by_log), min(SAMPLE, len(by_log)))
    sizes = []
    for index, log in enumerate(sample, 1):
        total = sum(head_size(p) for p in by_log[log])
        sizes.append(total)
        print(f"  {index:2d}/{len(sample)}  {total / 1e9:6.3f} GB  "
              f"{len(by_log[log]):2d} files  {log.split('/')[-1][:44]}")
        sys.stdout.flush()

    mean = statistics.mean(sizes)
    spread = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
    projected = mean * len(by_log)
    error = spread / len(sizes) ** 0.5 * len(by_log)
    print()
    print(f"mean log:  {mean / 1e9:.3f} GB   (sd {spread / 1e9:.3f} GB over {len(sizes)} logs)")
    print(f"smallest:  {min(sizes) / 1e9:.3f} GB    largest: {max(sizes) / 1e9:.3f} GB")
    print(f"PROJECTED TOTAL for {len(by_log)} logs: "
          f"{projected / 1e9:.1f} GB  +/- {error / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
