"""Fetch only the five pinned author source files; no installation or execution."""
from __future__ import annotations

import argparse
import json
import os
from urllib.request import Request, urlopen

from handoff_artifact_utils import ROOT, repository_path, sha256

COMMIT = "7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c"
FILES = {"tam.py", "demo.py", "qwen_utils.py", "requirements.txt", "README.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Explicitly permit downloading pinned source files")
    args = parser.parse_args()
    receipt_bytes = (ROOT / "handoff/tam_upstream_receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt["commit"] != COMMIT or receipt["repository"] != "https://github.com/xmed-lab/TAM" or set(receipt["SHA"]) != FILES:
        raise ValueError("Unexpected upstream binding")
    destination = repository_path(".tools/tam_official_v1")
    pending = []
    for name in sorted(FILES):
        file = destination / name
        if file.exists():
            if sha256(file.read_bytes()) != receipt["SHA"][name]:
                raise ValueError(f"Conflicting cached source preserved: {name}")
        else:
            pending.append(name)
    cached_receipt = destination / "source_receipt.json"
    if cached_receipt.exists() and cached_receipt.read_bytes() != receipt_bytes:
        raise ValueError("Conflicting cached source receipt preserved")
    if not args.fetch:
        print(f"Pinned TAM source: {COMMIT}; {len(FILES) - len(pending)} verified cached files, {len(pending)} missing.")
        print("Use --fetch to download source only into ignored .tools/tam_official_v1.")
        return
    # Validate all network payloads before creating any new cache file.
    downloads = {}
    for name in pending:
        url = f"https://raw.githubusercontent.com/xmed-lab/TAM/{COMMIT}/{name}"
        request = Request(url, headers={"User-Agent": "BehaviorICL-pinned-source-handoff"})
        with urlopen(request, timeout=30) as response:
            data = response.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024 or sha256(data) != receipt["SHA"][name]:
            raise ValueError(f"Downloaded upstream SHA/size mismatch: {name}")
        downloads[name] = data
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in downloads.items():
        with (destination / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    if not cached_receipt.exists():
        with cached_receipt.open("xb") as stream:
            stream.write(receipt_bytes)
    print(f"Verified all {len(FILES)} author source files at {COMMIT}. No install, model, or cloud operation performed.")


if __name__ == "__main__":
    main()
