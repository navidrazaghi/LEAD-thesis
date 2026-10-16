# -*- coding: utf-8 -*-
"""Page through the repository tree instead of trusting one capped response.

``fetch_listing`` asked ``/api/datasets/<repo>`` once and treated the
``siblings`` array as the whole repository. That field is capped. The cached
response holds 82,050 paths and stops part-way through the alphabet at
HazardAtSideLaneTwoWays, so fourteen of the release's forty-three scenario
types were candidates for selection and twenty-nine were not, and the second
camera view is absent from it entirely. The dataset card reports 8,930 logs in
`logs/normal_view`; the selector saw 2,610 of them.

Nothing failed. A truncated listing is a valid listing of a smaller repository,
and every count downstream was correct about the wrong input.

The tree endpoint pages, and says so in a Link header, so this follows it to
the end. An old cache is still read -- deleting it silently would refetch
several minutes of API calls on a run that only wanted the cached answer -- but
it is announced as capped, because a listing that looks complete and is not is
exactly what caused this.
"""

import io
import pathlib
import sys

SCRIPT = pathlib.Path.home() / "LEAD/lead/scripts/common/fetch_dataset_subset.py"

IMPORT_ANCHOR = "from xml.etree import ElementTree\n"
IMPORT_BLOCK = "from collections.abc import Iterator\nfrom xml.etree import ElementTree\n"

API_ANCHOR = 'API = f"https://huggingface.co/api/datasets/{REPO}"\n'
API_BLOCK = (
    'API = f"https://huggingface.co/api/datasets/{REPO}"\n'
    '# The repository listing. Unlike the dataset endpoint, whose "siblings"\n'
    "# array is capped, this one pages and reports the next page in a Link\n"
    "# header.\n"
    'TREE = f"{API}/tree/main?recursive=true"\n'
)

OLD = '''def fetch_listing(cache: pathlib.Path) -> list[str]:
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
    return [s["rfilename"] for s in payload.get("siblings", [])]'''

NEW = '''def next_page(header: str) -> str | None:
    """The URL of the next page of a tree listing, if the header names one.

    Args:
        header: The response's Link header, which may be empty.

    Returns:
        The next page's URL, or None at the end of the listing.
    """
    for part in header.split(","):
        match = re.search(r'<([^>]+)>;\\s*rel="next"', part)
        if match:
            return match.group(1)
    return None


def walk_tree(url: str) -> Iterator[str]:
    """Every file path in the repository, following the listing's pages.

    Args:
        url: The first page of the tree listing.

    Yields:
        One repository file path per entry, directories skipped.
    """
    page_number = 0
    while url:
        page_number += 1
        request = urllib.request.Request(url, headers={"User-Agent": "lead-fetch"})  # noqa: S310
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            entries = json.loads(response.read())
            link = response.headers.get("Link", "")
        for entry in entries:
            if entry.get("type") == "file":
                yield str(entry["path"])
        url = next_page(link) or ""
    print(f"  read {page_number} page(s) of the repository tree")


def fetch_listing(cache: pathlib.Path) -> list[str]:
    """The repo's file list, downloaded once and cached on disk.

    Args:
        cache: Where to keep the listing.

    Returns:
        Every file path in the repo.
    """
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # A response from the dataset endpoint, kept from before this
            # function paged. Its file list is capped, so a selection made on
            # it sees only the alphabetically first part of the repository.
            print(
                "  WARNING: the cached listing is a dataset-endpoint response, "
                "whose file list is capped. Delete it to fetch the full tree.",
            )
            return [s["rfilename"] for s in payload.get("siblings", [])]
        return [str(path) for path in payload]
    paths = list(walk_tree(TREE))
    cache.write_text(json.dumps(paths), encoding="utf-8")
    return paths'''

text = io.open(SCRIPT, encoding="utf-8").read()

if "def walk_tree(" in text:
    sys.exit("already patched; nothing to do")

for anchor in (IMPORT_ANCHOR, API_ANCHOR, OLD):
    if text.count(anchor) != 1:
        sys.exit(f"FATAL: anchor appears {text.count(anchor)} times, expected 1:\n"
                 f"{anchor[:90]}")

text = (
    text.replace(IMPORT_ANCHOR, IMPORT_BLOCK)
    .replace(API_ANCHOR, API_BLOCK)
    .replace(OLD, NEW)
)
io.open(SCRIPT, "w", encoding="utf-8", newline="\n").write(text)
print("patched fetch_dataset_subset.py")
