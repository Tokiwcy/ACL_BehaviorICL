"""Explicit human resume after a stopped worker/reboot; frozen science unchanged."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import internvl35_qwen_config_v4 as legacy
import internvl35_joint_config_v1 as joint
import run_internvl35_joint_pipeline_v1 as pipeline

RECEIPT = joint.ANALYSIS / "user_resume_20261008_v1.json"


def child(argv, log, status=None, record=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{joint.now()} {argv!r}\n")
        handle.flush()
        worker = subprocess.Popen(argv, cwd=joint.ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if record is not None:
            record.update(state="running", worker_pid=worker.pid, resumed_utc=joint.now())
            joint.write_json(legacy.OUTPUT / "pipeline_logs/status.json", status)
        result = worker.wait()
    if record is not None:
        record.update(state="complete" if result == 0 else "failed", return_code=result,
                      finished_utc=joint.now(), log=str(log))
        joint.write_json(legacy.OUTPUT / "pipeline_logs/status.json", status)
    if result:
        raise subprocess.CalledProcessError(result, argv)


def check():
    legacy.ensure_protocol()
    joint.ensure_protocol()
    status = json.loads((legacy.OUTPUT / "pipeline_logs/status.json").read_text(encoding="utf-8"))
    if status["state"] != "paused" or status["current"] != {"dataset": "aircraft", "stage": "icl_rices"}:
        raise RuntimeError("Resume applies only to the explicitly paused Aircraft RICES stage")
    if (joint.ANALYSIS / "compatibility.json").exists() or (pipeline.LOG_ROOT / "status.json").exists():
        raise RuntimeError("Joint GPU phase already exists; do not launch it twice")
    from run_internvl35_qwen_methods_v4 import load_context
    folder, bank, queries, identity = load_context("aircraft")
    selections = json.loads((folder / "selections/rices.json").read_text(encoding="utf-8"))
    if list(selections) != [q.sample_id for q in queries]:
        raise ValueError("Selection order changed")
    data = (folder / "predictions/rices.jsonl").read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("Uncommitted prediction tail; diagnose before resume")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != 3329:
        raise ValueError("This explicit operation is bound to the saved 3329-row pause")
    for i, row in enumerate(rows):
        q = queries[i]
        demos = [bank[j] for j in selections[q.sample_id]]
        if (row["query_id"], row["target"], row["demo_ids"], row["demo_labels"], row["correct"]) != (
            q.sample_id, q.label, [d.sample_id for d in demos], [d.label for d in demos], row["prediction"] == q.label):
            raise ValueError("Existing prediction identity changed")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    by_id = {r["sample_id"]: r for part in ("bank", "query") for r in manifest[part]}
    for target in queries[len(rows):]:
        for sample in [bank[j] for j in selections[target.sample_id]] + [target]:
            if legacy.digest(sample.path) != by_id[sample.sample_id]["image_sha256"]:
                raise ValueError("Remaining ICL image bytes changed")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="CPU-only validation, no model/CUDA calls")
    args = parser.parse_args()
    status = check()
    if args.check:
        print("Exact protocols, 3329-row prefix and remaining4 image SHA verified; CPU only", flush=True)
        return
    with pipeline.lock_transition(), pipeline.lock_gpu():
        pipeline.gpu_free()
        if RECEIPT.exists():
            raise RuntimeError("This explicit resume operation was already used; never repeat a consumed launcher")
        prior_transition = joint.ANALYSIS / "transition.json"
        backup = joint.ANALYSIS / "transition.pre_user_resume_20261008_v1.json"
        if backup.exists():
            raise RuntimeError("Resume backup already exists")
        backup.write_bytes(prior_transition.read_bytes())
        old = json.loads(backup.read_text(encoding="utf-8"))
        observed_boot = pipeline.boot_identity()
        receipt = {"state": "finishing_aircraft", "started_utc": joint.now(), "pid": os.getpid(),
                   "authorization": "Human explicitly resumed this task on2026-10-08 after user pause",
                   "previous_boot_identity": old["boot_identity"], "boot_identity": observed_boot,
                   "code_sha256": joint.digest(Path(__file__)),
                   "base_protocol_sha256": joint.digest(legacy.PROTOCOL_PATH),
                   "protocol_sha256": joint.digest(joint.PROTOCOL_PATH),
                   "saved_predictions": 3329, "remaining_predictions": 4,
                   "os_event_basis": "Actual boot2026-10-08T09:21:41.5Z, Event6005 at17:21:57+08; no41/1001/6008 in queried2h; cause not asserted"}
        joint.write_json(RECEIPT, receipt)
        original_boot = pipeline.boot_identity
        def operation_boot():
            from datetime import datetime
            parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
            current = original_boot()
            if abs((parse(current) - parse(observed_boot)).total_seconds()) > 10:
                raise RuntimeError("Boot identity changed during resumed operation; stop without retry")
            return observed_boot
        pipeline.boot_identity = operation_boot
        try:
            flag = pipeline.LOG_ROOT / "STOP_REQUESTED"
            if flag.exists():
                archived = flag.with_name("STOP_REQUESTED.user_pause_20261008_v1")
                if archived.exists():
                    raise RuntimeError("Pause marker already archived")
                flag.rename(archived)
            joint.write_json(prior_transition, receipt)
            status.setdefault("resume_history", []).append({"paused_status_sha256": joint.digest(joint.ANALYSIS / "v4_status.pre_user_pause_20261008.json"),
                                                              "resumed_utc": joint.now(), "operation": str(RECEIPT)})
            status.update(state="running", supervisor_pid=os.getpid(), checked_utc=joint.now())
            record = status["stages"][-1]
            child(record["command"], legacy.OUTPUT / "pipeline_logs/aircraft_icl_rices.log", status, record)
            status.update(state="paused", checked_utc=joint.now(), pause_reason="Aircraft complete; v4 stops before CUB joint handover")
            joint.write_json(legacy.OUTPUT / "pipeline_logs/status.json", status)
            child(pipeline.cmd("audit_internvl35_qwen_config_v4.py", "--allow-incomplete"), joint.ANALYSIS / "aircraft_audit.log")
            audit = json.loads((legacy.ANALYSIS / "progress_audit.json").read_text(encoding="utf-8"))
            if any(audit["datasets"]["aircraft"]["methods"][m]["state"] != "complete" for m in joint.METHODS):
                raise RuntimeError("Aircraft independent audit did not complete all six cells")
            receipt.update(state="compatibility", checked_utc=joint.now())
            joint.write_json(RECEIPT, receipt)
            joint.write_json(prior_transition, receipt)
            pipeline.gpu_free()
            child(pipeline.cmd("check_internvl35_joint_v1.py"), joint.ANALYSIS / "compatibility.log")
            receipt.update(state="running_joint_pipeline", checked_utc=joint.now())
            joint.write_json(RECEIPT, receipt)
            joint.write_json(prior_transition, receipt)
            pipeline.gpu_free()
            pipeline.pipeline(False)
            receipt.update(state="complete", finished_utc=joint.now())
            joint.write_json(RECEIPT, receipt)
            joint.write_json(prior_transition, receipt)
        except BaseException as error:
            receipt.update(state="failed", error=repr(error), finished_utc=joint.now())
            joint.write_json(RECEIPT, receipt)
            joint.write_json(prior_transition, receipt)
            if status.get("state") == "running":
                status.update(state="failed", finished_utc=joint.now())
                joint.write_json(legacy.OUTPUT / "pipeline_logs/status.json", status)
            raise

if __name__ == "__main__":
    main()
