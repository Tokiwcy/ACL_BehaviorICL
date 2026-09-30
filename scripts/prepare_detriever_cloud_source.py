#!/usr/bin/env python
"""Rebase an existing tuple's manifest paths after moving frozen states to Linux.

Only paths change. Every bank/query sample ID, label, split and order must
match the original manifest before the new DeTriever can use its cached input
states; this prevents an otherwise silent feature-to-image misalignment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from multidataset_protocol import DATASETS, load_dataset, run_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    args = parser.parse_args()
    run_dir = run_directory(args.source_root, args.dataset, "qwen3vl4b", args.seed)
    manifest_path = run_dir / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank, query = load_dataset(args.dataset, args.datasets_root)
    for name, current in (("bank", bank), ("query", query)):
        old = [(row["sample_id"], row["label"], row["split"]) for row in original[name]]
        new = [(row.sample_id, row.label, row.split) for row in current]
        if old != new:
            raise RuntimeError(f"{name} order/identity mismatch; refusing to rebase cached states")
    progress = json.loads((run_dir / "anchor_states_progress.json").read_text(encoding="utf-8"))
    if progress["completed"] != len(bank) + len(query):
        raise RuntimeError("Frozen input-state cache is incomplete")
    target = {"bank": [asdict(row) for row in bank], "query": [asdict(row) for row in query]}
    manifest_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rebased {args.dataset}: bank={len(bank)}, query={len(query)}; IDs/labels/order unchanged")


if __name__ == "__main__":
    main()
