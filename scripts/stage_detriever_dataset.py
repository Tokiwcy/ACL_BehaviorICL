#!/usr/bin/env python
"""Verify and extract one pinned local dataset archive on the cloud Pod."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from multidataset_protocol import load_dataset


ARCHIVES = {
    "aircraft": [
        ("fgvc-aircraft-2013b.tar.gz", "e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834"),
    ],
    "pets": [
        ("oxford-pets-images.tar.gz", "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d"),
        ("oxford-pets-annotations.tar.gz", "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91"),
    ],
    "cub": [
        ("CUB_200_2011.tgz", "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"),
    ],
    "dogs": [
        ("stanford-dogs-images-mirror.zip", "d7e7c4d08d0df25964f9a2835e0f91fcfafd3d26c6ad083cd82d7cece4be42b4"),
        ("stanford-dogs-lists.tar", "34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c"),
    ],
}

EXPECTED = {
    "aircraft": (6667, 3333, 100),
    "pets": (3680, 3669, 37),
    "cub": (5994, 5794, 200),
    "dogs": (12000, 8580, 120),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ARCHIVES, required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    root = args.datasets_root
    ready = root / f".{args.dataset}-detriever-ready.json"
    if ready.exists():
        saved = json.loads(ready.read_text(encoding="utf-8"))
        if saved.get("expected") != list(EXPECTED[args.dataset]):
            raise RuntimeError("Existing readiness marker has a different protocol")
        print(f"{args.dataset} already staged: {saved['expected']}", flush=True)
        return
    paths = []
    for name, expected_hash in ARCHIVES[args.dataset]:
        path = root / "_archives" / name
        if not path.is_file() or digest(path) != expected_hash:
            raise RuntimeError(f"Missing or checksum-mismatched archive: {path}")
        paths.append(path)
    if args.dataset == "aircraft":
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(paths[0]), "-C", str(root)], check=True)
    elif args.dataset == "pets":
        target = root / "oxford_iiit_pet"
        target.mkdir(parents=True, exist_ok=True)
        for path in paths:
            subprocess.run(["tar", "--no-same-owner", "-xzf", str(path), "-C", str(target)], check=True)
    elif args.dataset == "cub":
        subprocess.run(["tar", "--no-same-owner", "-xzf", str(paths[0]), "-C", str(root)], check=True)
    else:
        target = root / "stanford_dogs"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(paths[0]) as archive:
            archive.extractall(target)
        extracted = target / "stanford_dog_dataset"
        images = target / "Images"
        if extracted.is_dir() and not images.exists():
            extracted.rename(images)
        subprocess.run(["tar", "--no-same-owner", "-xf", str(paths[1]), "-C", str(target)], check=True)
    bank, query = load_dataset(args.dataset, root)
    actual = (len(bank), len(query), len({row.label for row in bank}))
    if actual != EXPECTED[args.dataset]:
        raise RuntimeError(f"Unexpected {args.dataset} split: {actual}")
    ready.write_text(json.dumps({"dataset": args.dataset, "expected": list(actual),
                                 "archive_sha256": {p.name: digest(p) for p in paths}}, indent=2),
                     encoding="utf-8")
    print(f"{args.dataset} staged: bank={actual[0]} test={actual[1]} classes={actual[2]}", flush=True)


if __name__ == "__main__":
    main()
