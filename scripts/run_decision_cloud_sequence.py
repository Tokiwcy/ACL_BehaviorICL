#!/usr/bin/env python
"""Run Behavior-Zero/Learn tuples serially, retaining legacy artifact keys."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ORDER = ("pets", "cub", "dogs")


def write_progress(path: Path, dataset: str, stage: str, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "stage": stage,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def source_count(root: Path, dataset: str, source_root: Path) -> tuple[int, int]:
    source = root / source_root / dataset / "qwen3vl4b/seed_73"
    identity = json.loads((source / "run_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((source / "anchor_states_progress.json").read_text(encoding="utf-8"))
    bank, query = manifest["bank"], manifest["query"]
    expected = len(bank) + len(query)
    if {key: identity.get(key) for key in ("dataset", "model_slug", "seed")} != {
        "dataset": dataset, "model_slug": "qwen3vl4b", "seed": 73,
    }:
        raise RuntimeError(f"Frozen state identity mismatch: {source}")
    if progress.get("completed") != expected or progress.get("shape") != [expected, 36, 2560]:
        raise RuntimeError(f"Frozen state cache incomplete: {source}")
    if not (source / "anchor_states.npy").is_file():
        raise RuntimeError(f"Frozen state file missing: {source}")
    return len(bank), len(query)


def verify(root: Path, dataset: str, source_root: Path, output_root: Path) -> None:
    run_dir = root / output_root / dataset / "qwen3vl4b/seed_73"
    source_dir = root / source_root / dataset / "qwen3vl4b/seed_73"
    subprocess.run(
        [sys.executable, "scripts/verify_dtd_decision_states.py", str(run_dir),
         "--source-run", str(source_dir)],
        cwd=root, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/BehaviorICL"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--output-root", type=Path, default=Path("results/decision_main"))
    parser.add_argument("--start-at", choices=ORDER, default="pets")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.generation_batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    root = args.root.resolve()
    progress_path = root / args.output_root / "sequence_progress.json"
    start = ORDER.index(args.start_at)
    current = args.start_at
    try:
        for dataset in ORDER[:start]:
            current = dataset
            verify(root, dataset, args.source_root, args.output_root)
        for dataset in ORDER[start:]:
            current = dataset
            write_progress(progress_path, dataset, "staging_dataset")
            subprocess.run(
                [sys.executable, "scripts/stage_detriever_dataset.py",
                 "--dataset", dataset, "--datasets-root", str(root / "datasets")],
                cwd=root, check=True,
            )
            write_progress(progress_path, dataset, "extracting_states")
            subprocess.run(
                [sys.executable, "-u", "scripts/prepare_detriever_input_states.py",
                 "--dataset", dataset, "--seed", "73",
                 "--datasets-root", str(root / "datasets"),
                 "--output-root", str(root / args.source_root),
                 "--cache-dir", str(root / ".hf_cache/hub"),
                 "--batch-size", "1", "--resume"],
                cwd=root, check=True,
            )
            bank, query = source_count(root, dataset, args.source_root)
            write_progress(progress_path, dataset, "running", bank=bank, query=query)
            subprocess.run(
                [sys.executable, "-u", "scripts/run_dtd_decision_states.py",
                 "--dataset", dataset, "--datasets-root", str(root / "datasets"),
                 "--source-root", str(root / args.source_root),
                 "--output-root", str(root / args.output_root),
                 "--cache-dir", str(root / ".hf_cache/hub"),
                 "--seed", "73", "--shots", "4", "--train-steps", "1500",
                 "--generation-batch-size", str(args.generation_batch_size), "--resume"],
                cwd=root, check=True,
            )
            write_progress(progress_path, dataset, "verifying", bank=bank, query=query)
            verify(root, dataset, args.source_root, args.output_root)
            write_progress(progress_path, dataset, "complete_verified", bank=bank, query=query)
        write_progress(progress_path, ORDER[-1], "all_complete_verified", order=list(ORDER))
    except Exception as error:
        write_progress(progress_path, current, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
