"""Rebuild host-specific manifest paths before a generation-only cloud resume."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from multidataset_protocol import load_dataset
from run_multidataset_main import METHODS, validate_generation_resume, write_json


def main() -> None:
    run_dir = Path("results/cdr_main/aircraft/qwen3vl4b/seed_73")
    identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    expected_identity = {
        "dataset": "aircraft",
        "model_slug": "qwen3vl4b",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "seed": 73,
    }
    if identity != expected_identity:
        raise RuntimeError("Refusing to prepare a different experiment tuple")

    bank, query = load_dataset("aircraft", Path("datasets"))
    if (len(bank), len(query)) != (6667, 3333):
        raise RuntimeError("Unexpected Aircraft split size")
    query_by_id = {sample.sample_id: sample for sample in query}
    labels = {sample.label for sample in bank}
    seen: set[str] = set()
    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = row["query_id"]
            if (
                row["method"] != "rices"
                or query_id in seen
                or query_id not in query_by_id
                or row["target"] != query_by_id[query_id].label
                or row["prediction"] not in labels
            ):
                raise RuntimeError("RICES prediction does not match this Aircraft split")
            seen.add(query_id)
    if seen != set(query_by_id):
        raise RuntimeError("RICES predictions are not complete")

    write_json(run_dir / "manifest.json", {
        "bank": [asdict(sample) for sample in bank],
        "query": [asdict(sample) for sample in query],
    })
    validate_generation_resume(run_dir, bank, query, list(METHODS), 4)
    print(f"Validated {len(bank)} bank, {len(query)} query, 5 selections, and {len(seen)} RICES predictions")


if __name__ == "__main__":
    main()
