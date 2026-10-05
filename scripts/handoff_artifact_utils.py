"""Portable, offline artifact checks. No model or cloud imports."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 256 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repository_path(relative: str) -> Path:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Unsafe repository path: {relative}")
    target = ROOT.joinpath(*path.parts)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path leaves repository: {relative}")
    return target


def manifest() -> dict:
    value = json.loads((ROOT / "handoff/publish_manifest.json").read_text(encoding="utf-8-sig"))
    names = [entry["path"] for entry in value["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate artifact paths")
    return value


def verified_payload(entry: dict) -> bytes:
    source = repository_path(entry["path"])
    if source.stat().st_size != entry["bytes"]:
        raise ValueError(f"Published byte length mismatch: {entry['path']}")
    data = source.read_bytes()
    if sha256(data) != entry["sha256"]:
        raise ValueError(f"Published SHA mismatch: {entry['path']}")
    if entry["kind"] == "exact":
        return data
    if entry["kind"] != "gzip_exact":
        raise ValueError(f"Unknown artifact kind: {entry['kind']}")
    limit = entry["source_bytes"]
    if not isinstance(limit, int) or not 0 <= limit <= MAX_SOURCE_BYTES:
        raise ValueError("Invalid decompression bound")
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as compressed:
        original = compressed.read(limit + 1)
        if len(original) != limit or compressed.read(1):
            raise ValueError(f"Uncompressed length mismatch: {entry['path']}")
    if sha256(original) != entry["source_sha256"]:
        raise ValueError(f"Original SHA mismatch: {entry['path']}")
    return original
