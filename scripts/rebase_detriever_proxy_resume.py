#!/usr/bin/env python
"""Rebase a partly extracted gold-proxy cache after verified Windows→Linux paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("proxy_run", type=Path)
    args = parser.parse_args()
    original_path = args.source_run / "manifest_windows.json"
    current_path = args.source_run / "manifest.json"
    before = json.loads(original_path.read_text(encoding="utf-8"))
    after = json.loads(current_path.read_text(encoding="utf-8"))
    for split in ("bank", "query"):
        old = [(x["sample_id"], x["label"], x["split"]) for x in before[split]]
        new = [(x["sample_id"], x["label"], x["split"]) for x in after[split]]
        if old != new:
            raise RuntimeError(f"{split} identities differ; refusing proxy-cache migration")
    old_hash, new_hash = digest(original_path), digest(current_path)
    identity_path = args.proxy_run / "run_identity.json"
    progress_path = args.proxy_run / "gold_input_answer_eos_progress.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if identity.get("baseline") != "detriever_gold_input_answer_eos_v1":
        raise RuntimeError("Unexpected retriever baseline identity")
    if identity.get("manifest_sha256") != old_hash or progress.get("manifest_sha256") != old_hash:
        raise RuntimeError("Local proxy cache hash does not match the original source manifest")
    if progress.get("shape", [None])[0] != len(after["bank"]):
        raise RuntimeError("Proxy cache shape does not match bank size")
    if not 0 <= progress.get("completed", -1) <= len(after["bank"]):
        raise RuntimeError("Invalid proxy completion count")
    identity["manifest_sha256"] = new_hash
    progress["manifest_sha256"] = new_hash
    write_json(identity_path, identity)
    write_json(progress_path, progress)
    write_json(args.proxy_run / "path_migration.json", {
        "original_manifest_sha256": old_hash,
        "rebased_manifest_sha256": new_hash,
        "completed_bank_proxy_rows": progress["completed"],
    })
    print(f"Rebased {progress['completed']}/{len(after['bank'])} bank gold-proxy rows")


if __name__ == "__main__":
    main()
