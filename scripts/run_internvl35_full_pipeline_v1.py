"""Resume the complete local InternVL3.5-4B five-dataset study.

Each scientific command runs in its own process on the one local GPU. The
underlying stage scripts own their checkpoints and verify the frozen manifest.
This supervisor records the exact command and return code for every stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dtd", "aircraft", "cub", "dogs", "pets")
METHODS = ("raw", "proj", "learn", "gpt_mm", "detriever", "rices")
RUN_ROOT = ROOT / "results/internvl35_full_v1"
LOG_ROOT = RUN_ROOT / "pipeline_logs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(value: dict) -> None:
    path = LOG_ROOT / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def command(script: str, dataset: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(ROOT / "scripts" / script), "--dataset", dataset, *args]


def stages(dataset: str):
    source = "run_internvl35_full_sources_v1.py"
    methods = "run_internvl35_full_methods_v1.py"
    icl = "run_internvl35_full_icl_v1.py"
    yield "anchor", command(source, dataset, "--stage", "anchor", "--batch-size", "1")
    yield "train_proj", command(methods, dataset, "--stage", "train_proj")
    yield "train_learn", command(methods, dataset, "--stage", "train_learn")
    yield "gpt_mm_source", command(source, dataset, "--stage", "gpt_mm", "--batch-size", "1")
    yield "detriever_proxy", command(methods, dataset, "--stage", "proxy")
    yield "train_detriever", command(methods, dataset, "--stage", "train_detriever")
    yield "clip_source", command(source, dataset, "--stage", "clip", "--batch-size", "32")
    for method in METHODS:
        yield f"retrieve_{method}", command(methods, dataset, "--stage", "retrieve", "--method", method)
    # DTD batch 4 was actually smoke-tested; longer label lists use batch 1.
    batch = "4" if dataset == "dtd" else "1"
    for method in METHODS:
        yield f"icl_{method}", command(icl, dataset, "--method", method, "--batch-size", batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-dtd-raw", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    planned = [(dataset, stage, argv) for dataset in DATASETS
               for stage, argv in stages(dataset)]
    status_path = LOG_ROOT / "status.json"
    if args.resume:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "paused":
            raise ValueError("Resume requires a paused pipeline")
        records = status["stages"]
        complete = [record for record in records if record["state"] == "complete"]
        if len(records) - len(complete) != 1 or records[-1]["state"] != "paused":
            raise ValueError("Expected completed prefix followed by one paused stage")
        for record, (dataset, stage, argv) in zip(complete, planned):
            if (record["dataset"], record["stage"], record["command"],
                    record.get("return_code")) != (dataset, stage, argv, 0):
                raise ValueError("Completed stage differs from current pipeline plan")
        next_index = len(complete)
        if next_index >= len(planned):
            raise ValueError("No remaining stages")
        paused = records[-1]
        if (paused["dataset"], paused["stage"], paused["command"]) != planned[next_index]:
            raise ValueError("Paused stage differs from current pipeline plan")
    else:
        if status_path.exists():
            raise FileExistsError("Existing pipeline status; use --resume")
        status = {"version": 1, "started_utc": now(), "state": "running", "stages": []}
        next_index = 0
    if args.dry_run:
        print(json.dumps({"resume": args.resume, "completed_stages": next_index,
                          "remaining_stages": len(planned) - next_index,
                          "next": planned[next_index][:2]}, ensure_ascii=False))
        return
    status["state"] = "running"
    if args.resume:
        status["resumed_utc"] = now()
    write_status(status)
    if args.wait_for_dtd_raw:
        import time

        summary = RUN_ROOT / "dtd/predictions/raw.summary.json"
        while not summary.is_file():
            status.update(state="waiting_for_dtd_raw", checked_utc=now())
            write_status(status)
            time.sleep(30)
        status["state"] = "running"
        write_status(status)
    for dataset, stage, argv in planned[next_index:]:
            record = {"dataset": dataset, "stage": stage, "command": argv,
                      "started_utc": now(), "state": "running"}
            status["stages"].append(record)
            status["current"] = {"dataset": dataset, "stage": stage}
            write_status(status)
            log_path = LOG_ROOT / f"{dataset}_{stage}.log"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n=== {record['started_utc']} {argv!r} ===\n")
                handle.flush()
                result = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            record.update(finished_utc=now(), return_code=result.returncode,
                          state="complete" if result.returncode == 0 else "failed",
                          log=str(log_path))
            write_status(status)
            if result.returncode != 0:
                status.update(state="failed", finished_utc=now())
                write_status(status)
                raise SystemExit(result.returncode)
    status.update(state="complete", finished_utc=now())
    status.pop("current", None)
    write_status(status)


if __name__ == "__main__":
    main()
