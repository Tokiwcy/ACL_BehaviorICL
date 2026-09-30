#!/usr/bin/env python
"""Serially finish the five Qwen gold-output DeTriever cloud tuples.

One process invokes at most one GPU child at a time. Archives are supplied by
the supervising host and checked before each new dataset starts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ORDER = ("dtd", "aircraft", "pets", "cub", "dogs")
ARCHIVE_SIZES = {
    "aircraft": {"fgvc-aircraft-2013b.tar.gz": 2753340328},
    "pets": {"oxford-pets-images.tar.gz": 791918971,
             "oxford-pets-annotations.tar.gz": 19173078},
    "cub": {"CUB_200_2011.tgz": 1150585339},
    "dogs": {"stanford-dogs-images-mirror.zip": 786911371,
             "stanford-dogs-lists.tar": 481280},
}
ROOT = Path("/workspace/BehaviorICL")
PYTHON = ROOT / ".venv/bin/python"
PROGRESS = ROOT / "results/detriever_output_proxy/sequence_progress.json"


def record(dataset: str, stage: str, **extra) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    value = {"dataset": dataset, "stage": stage,
             "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    PROGRESS.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def run(*args: str) -> None:
    subprocess.run([str(PYTHON), *args], cwd=ROOT, check=True)


def wait_for_dtd(pid: int) -> None:
    result = ROOT / "results/detriever_output_proxy/dtd/qwen3vl4b/seed_73"
    deadline = time.monotonic() + 2 * 3600
    record("dtd", "waiting_for_existing_run", pid=pid)
    while time.monotonic() < deadline:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "args="],
                                 capture_output=True, text=True, check=False).stdout
        active = "run_detriever_output_proxy.py --dataset dtd" in command
        if not active:
            if not (result / "summary.json").is_file():
                raise RuntimeError("DTD process ended without summary.json")
            break
        time.sleep(15)
    else:
        raise TimeoutError("DTD run exceeded two hours; supervisor must inspect")
    run("scripts/verify_detriever_output_proxy.py", str(result))
    record("dtd", "complete_verified")


def wait_for_archives(dataset: str) -> None:
    deadline = time.monotonic() + 45 * 60
    required = ARCHIVE_SIZES[dataset]
    while time.monotonic() < deadline:
        if all((ROOT / "datasets/_archives" / name).is_file()
               and (ROOT / "datasets/_archives" / name).stat().st_size == size
               for name, size in required.items()):
            return
        record(dataset, "waiting_for_archives", files=list(required))
        time.sleep(60)
    raise TimeoutError(f"{dataset} archives did not arrive within 45 minutes")


def finish_dataset(dataset: str, generation_batch_size: int) -> None:
    record(dataset, "waiting_for_archives")
    wait_for_archives(dataset)
    record(dataset, "verifying_and_extracting")
    run("scripts/stage_detriever_dataset.py", "--dataset", dataset)
    source = ROOT / f"results/cdr_main/{dataset}/qwen3vl4b/seed_73"
    if dataset == "aircraft":
        record(dataset, "rebasing_frozen_input_states")
        run("scripts/prepare_detriever_cloud_source.py", "--dataset", dataset)
    else:
        record(dataset, "extracting_frozen_input_states")
        command = ["scripts/prepare_detriever_input_states.py", "--dataset", dataset,
                   "--cache-dir", str(ROOT / ".hf_cache")]
        if (source / "run_identity.json").exists():
            command.append("--resume")
        run(*command)
    result = ROOT / f"results/detriever_output_proxy/{dataset}/qwen3vl4b/seed_73"
    record(dataset, "gold_proxy_train_and_generate")
    command = ["scripts/run_detriever_output_proxy.py", "--dataset", dataset,
               "--stage", "all", "--cache-dir", str(ROOT / ".hf_cache"),
               "--generation-batch-size", str(generation_batch_size)]
    if (result / "run_identity.json").exists():
        command.append("--resume")
    run(*command)
    record(dataset, "verifying")
    run("scripts/verify_detriever_output_proxy.py", str(result))
    record(dataset, "complete_verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-dtd-pid", type=int)
    parser.add_argument("--start-at", choices=ORDER[1:], default=None,
                        help="Resume serial execution at an existing dataset after a failure")
    parser.add_argument("--generation-batch-size", type=int, default=4)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT.resolve():
        raise RuntimeError(f"Run from {ROOT}")
    if args.generation_batch_size < 1:
        raise ValueError("Generation batch size must be positive")
    if args.start_at is None and args.wait_for_dtd_pid is None:
        raise ValueError("Initial sequence requires --wait-for-dtd-pid")
    try:
        if args.start_at is None:
            wait_for_dtd(args.wait_for_dtd_pid)
        for dataset in ORDER[ORDER.index(args.start_at) if args.start_at else 1:]:
            finish_dataset(dataset, args.generation_batch_size)
        record("dogs", "all_complete_verified", order=list(ORDER))
    except Exception as error:
        record(PROGRESS.stem, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
