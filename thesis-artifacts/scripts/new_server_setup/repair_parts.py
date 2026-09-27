"""Repair part files damaged by curl's --retry writing into an appended stream.

The second attempt asked curl for an explicit byte range and appended the
result to a part file, with --retry left on for robustness. Those two do not
compose. When a transfer dropped, --retry re-requested the *whole* range from
its first byte, and the shell appended that second pass after the bytes already
written. Every drop therefore left a duplicated prefix inside the part, and the
four parts together came to 9,362,519,223 bytes for a file of 8,346,095,504.

A damaged part is a sequence of attempts, each starting again at the range's
first byte, so the last attempt's data runs from its own start to the end of the
file. Finding where the last attempt began is enough to recover the part: the
first bytes of the file are that attempt's signature, and the last place they
appear is where it began. A part that then measures longer than its range is one
whose last attempt ran to completion, so its final `want` bytes are the range.

Nothing here is trusted on its own: the caller joins the repaired parts and runs
gzip -t over the result, which checks a CRC over all 8 GB and is the real proof.
"""
import pathlib
import sys

TOTAL = 8_346_095_504
PARTS = 4
SIGNATURE = 4096  # long enough that a coincidental match is not a real risk
WINDOW = 64 << 20


def last_occurrence(path: pathlib.Path, needle: bytes) -> int:
    """Offset of the last occurrence of `needle`, or 0 if it appears only once."""
    size = path.stat().st_size
    found = 0
    with path.open("rb") as handle:
        position = 0
        carry = b""
        while position < size:
            handle.seek(position)
            block = handle.read(WINDOW)
            if not block:
                break
            haystack = carry + block
            base = position - len(carry)
            index = haystack.rfind(needle)
            if index != -1 and base + index > 0:
                found = base + index
            carry = haystack[-(len(needle) - 1):]
            position += len(block)
    return found


def main() -> None:
    directory = pathlib.Path(sys.argv[1])
    chunk = TOTAL // PARTS
    for index in range(PARTS):
        part = directory / f"CARLA_0916.tar.gz.part{index}"
        if not part.exists():
            print(f"part{index}: missing, will be downloaded fresh")
            continue
        start = index * chunk
        end = TOTAL - 1 if index == PARTS - 1 else start + chunk - 1
        want = end - start + 1
        have = part.stat().st_size
        if have <= want:
            # Still possible for a short part to carry a restart inside it.
            with part.open("rb") as handle:
                needle = handle.read(SIGNATURE)
            offset = last_occurrence(part, needle)
            if offset == 0:
                print(f"part{index}: {have:,} of {want:,} bytes, clean")
                continue
        else:
            with part.open("rb") as handle:
                needle = handle.read(SIGNATURE)
            offset = last_occurrence(part, needle)

        keep = have - offset
        if keep > want:
            # The last attempt completed, so the range is its final `want` bytes.
            offset = have - want
            keep = want
        print(f"part{index}: {have:,} bytes -> keeping the last {keep:,} "
              f"from offset {offset:,} (range is {want:,})")
        repaired = part.with_suffix(part.suffix + ".fixed")
        with part.open("rb") as source, repaired.open("wb") as target:
            source.seek(offset)
            while True:
                block = source.read(WINDOW)
                if not block:
                    break
                target.write(block)
        repaired.replace(part)
        print(f"part{index}: now {part.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
