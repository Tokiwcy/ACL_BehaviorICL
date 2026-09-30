#!/usr/bin/env python
"""Print a deterministic byte-level digest of the FGVC-Aircraft data tree."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from multidataset_protocol import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    overall = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        overall.update(relative.encode("utf-8"))
        overall.update(b"\0")
        overall.update(digest.digest())
    bank, query = load_dataset("aircraft", root.parent.parent)
    print(
        f"files={len(files)} bytes={total_bytes} sha256={overall.hexdigest()} "
        f"bank={len(bank)} query={len(query)} labels={len({row.label for row in bank})}"
    )


if __name__ == "__main__":
    main()
