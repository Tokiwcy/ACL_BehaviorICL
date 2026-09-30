#!/usr/bin/env python
"""Queue the DTD state ablation behind the single-GPU four-method sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def record(path: Path, stage: str, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/BehaviorICL"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.poll_seconds < 10:
        raise ValueError("Poll interval must be at least ten seconds")
    sequence = root / "results/cdr_main/four_method_sequence_progress.json"
    progress = root / "results/decision_state_dtd/queue_progress.json"
    result = root / "results/decision_state_dtd/dtd/qwen3vl4b/seed_73"
    record(progress, "waiting_for_four_method_sequence")
    try:
        while True:
            if sequence.is_file():
                state = json.loads(sequence.read_text(encoding="utf-8"))
                if state.get("stage") == "failed":
                    raise RuntimeError("Four-method sequence failed; refusing concurrent DTD run")
                if state.get("stage") == "all_complete_verified":
                    if state.get("order") != ["pets", "cub", "dogs"]:
                        raise RuntimeError("Four-method completion marker has unexpected order")
                    break
            time.sleep(args.poll_seconds)
        # The completion marker is written at the very end of the serial process.
        # Wait for its process to exit before starting another GPU workload.
        for _ in range(30):
            active = subprocess.run(
                ["pgrep", "-af", "scripts/run_multidataset_main.py"],
                capture_output=True, text=True, check=False,
            )
            if not any("python" in line and "run_multidataset_main.py --dataset" in line
                       for line in active.stdout.splitlines()):
                break
            time.sleep(2)
        else:
            raise RuntimeError("A four-method GPU child remained active after completion")
        record(progress, "decision_running", source="results/cdr_main/dtd/qwen3vl4b/seed_73")
        command = [sys.executable, "-u", "scripts/run_dtd_decision_states.py",
                   "--cache-dir", str(root / ".hf_cache"),
                   "--generation-batch-size", "1"]
        if (result / "run_identity.json").is_file():
            command.append("--resume")
        subprocess.run(command, cwd=root, check=True)
        record(progress, "decision_verifying")
        subprocess.run([sys.executable, "scripts/verify_dtd_decision_states.py", str(result)],
                       cwd=root, check=True)
        record(progress, "decision_complete_verified")
    except Exception as error:
        record(progress, "failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
