#!/usr/bin/env python
"""Run the four non-DeTriever Qwen methods serially on prepared cloud tuples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ORDER = ("aircraft", "pets", "cub", "dogs")
METHODS = ("rices", "gpt_mm", "cdr_zero", "cdr_learn")


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


def run_dataset(root: Path, dataset: str, batch_size: int, progress: Path) -> None:
    run_dir = root / "results/cdr_main" / dataset / "qwen3vl4b/seed_73"
    identity_path = run_dir / "run_identity.json"
    manifest_path = run_dir / "manifest.json"
    state_progress_path = run_dir / "anchor_states_progress.json"
    if not all(path.is_file() for path in (identity_path, manifest_path, state_progress_path)):
        raise RuntimeError(f"Frozen source tuple is incomplete: {run_dir}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state_progress = json.loads(state_progress_path.read_text(encoding="utf-8"))
    expected = len(manifest["bank"]) + len(manifest["query"])
    if identity["dataset"] != dataset or identity["model_slug"] != "qwen3vl4b" or identity["seed"] != 73:
        raise RuntimeError(f"Frozen source identity mismatch: {run_dir}")
    if state_progress["completed"] != expected or not (run_dir / "anchor_states.npy").is_file():
        raise RuntimeError(f"Frozen source cache incomplete: {run_dir}")
    config_path = run_dir / "config.json"
    if config_path.is_file():
        existing_methods = set(json.loads(config_path.read_text(encoding="utf-8")).get("methods", []))
        if existing_methods != set(METHODS):
            raise RuntimeError(f"Existing run config is not the four-method protocol: {run_dir}")
    elif any((run_dir / name).exists() for name in
             ("selections.json", "predictions.jsonl", "projected_velocity_best.pt")):
        raise RuntimeError(f"Found old selections, predictions, or training in frozen source: {run_dir}")

    write_progress(progress, dataset, "running", bank=len(manifest["bank"]), query=len(manifest["query"]))
    command = [
        sys.executable, "-u", "scripts/run_multidataset_main.py",
        "--dataset", dataset, "--model", "qwen3vl4b", "--seed", "73",
        "--shots", "4", "--cache-dir", str(root / ".hf_cache"),
        "--output-root", "results/cdr_main", "--stage", "all", "--resume",
        "--generation-batch-size", str(batch_size), "--extract-batch-size", "1",
        "--methods", *METHODS,
    ]
    subprocess.run(command, cwd=root, check=True)
    write_progress(progress, dataset, "verifying")
    subprocess.run([sys.executable, "scripts/verify_main_run.py", str(run_dir),
                    "--expected-methods", *METHODS], cwd=root, check=True)
    write_progress(progress, dataset, "complete_verified", query=len(manifest["query"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/BehaviorICL"))
    parser.add_argument("--datasets", nargs="+", choices=ORDER, required=True)
    parser.add_argument("--start-at", choices=ORDER, default=None,
                        help="Resume at this dataset after verifying all preceding datasets")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.generation_batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Dataset sequence contains duplicates")
    if list(args.datasets) != [name for name in ORDER if name in args.datasets]:
        raise ValueError("Dataset sequence must follow Aircraft, Pets, CUB, Dogs order")
    if args.start_at is not None and args.start_at not in args.datasets:
        raise ValueError("--start-at must be included in --datasets")
    root = args.root.resolve()
    progress = root / "results/cdr_main/four_method_sequence_progress.json"
    start_index = args.datasets.index(args.start_at) if args.start_at is not None else 0
    current_dataset = args.datasets[start_index]
    try:
        for dataset in args.datasets[:start_index]:
            current_dataset = dataset
            run_dir = root / "results/cdr_main" / dataset / "qwen3vl4b/seed_73"
            subprocess.run([sys.executable, "scripts/verify_main_run.py", str(run_dir),
                            "--expected-methods", *METHODS], cwd=root, check=True)
        for dataset in args.datasets[start_index:]:
            current_dataset = dataset
            run_dataset(root, dataset, args.generation_batch_size, progress)
        write_progress(progress, args.datasets[-1], "all_complete_verified", order=args.datasets)
    except Exception as error:
        write_progress(progress, current_dataset, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
