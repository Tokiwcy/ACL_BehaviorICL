#!/usr/bin/env python
"""Run Aircraft Decision and forward-trace ablations serially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ("decision", "run_dtd_decision_states.py", "results/decision_state_aircraft"),
    ("forward_trace", "run_dtd_forward_trace.py", "results/full_forward_aircraft"),
)


def write_progress(stage: str, status: str, error: str | None = None) -> None:
    path = ROOT / "results" / "aircraft_process_ablation" / "sequence_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage, "status": status, "error": error},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-anchor", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if args.prepare_anchor:
        write_progress("anchor", "running")
        source = [sys.executable, str(ROOT / "scripts" / "prepare_aircraft_anchor_source.py"),
                  "--resume"]
        result = subprocess.run(source, cwd=ROOT, env=env, check=False)
        if result.returncode:
            write_progress("anchor", "failed", f"exit_code={result.returncode}")
            raise SystemExit(result.returncode)
        write_progress("anchor", "complete")
    for name, script, output_root in STAGES:
        run_dir = ROOT / output_root / "aircraft" / "qwen3vl4b" / "seed_73"
        summary = run_dir / "summary.json"
        if summary.is_file():
            data = json.loads(summary.read_text(encoding="utf-8"))["summary"]
            if set(data) == ({"decision_zero", "decision_learn"} if name == "decision"
                             else {"forward_zero", "forward_learn"}) and all(
                    row["total"] == 3333 for row in data.values()):
                print(f"{name} already has a complete summary; checking it", flush=True)
                rerun = False
            else:
                rerun = True
        else:
            rerun = True
        if rerun:
            write_progress(name, "running")
            command = [sys.executable, str(ROOT / "scripts" / script), "--dataset", "aircraft",
                       "--output-root", str(ROOT / output_root), "--resume"]
            print(f"Starting {name}: {' '.join(command)}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if result.returncode:
                write_progress(name, "failed", f"exit_code={result.returncode}")
                raise SystemExit(result.returncode)
        verifier = "verify_dtd_decision_states.py" if name == "decision" else "verify_dtd_forward_trace.py"
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / verifier), str(run_dir)],
                                cwd=ROOT, env=env, check=False)
        if result.returncode:
            write_progress(name, "failed_verification", f"exit_code={result.returncode}")
            raise SystemExit(result.returncode)
        write_progress(name, "complete_verified")
    write_progress("all", "complete_verified")
    print("Aircraft Decision and forward-trace generation complete and verified", flush=True)


if __name__ == "__main__":
    main()
