"""Graceful Aircraft boundary handover, one compatibility gate, sequential GPU."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from internvl35_joint_config_v1 import *
import internvl35_qwen_config_v4 as legacy
from run_internvl35_qwen_pipeline_v4 import lock_gpu, gpu_free

LOG_ROOT = OUTPUT / "pipeline_logs"

def lock_transition():
    import msvcrt
    OUTPUT.mkdir(parents=True, exist_ok=True)
    handle = (OUTPUT / "transition.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        raise RuntimeError("Another joint transition/supervisor already exists")
    return handle

def cmd(script, *args):
    return [sys.executable, "-u", str(ROOT / "scripts" / script), *args]


def prepare():
    ensure_protocol()
    from multidataset_protocol import load_dataset
    from dataclasses import asdict
    for dataset in DATASETS:
        source = legacy.OUTPUT / dataset
        folder = OUTPUT / dataset
        folder.mkdir(parents=True, exist_ok=True)
        bank, query = load_dataset(dataset, ROOT / "datasets")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        for key, samples in (("bank", bank), ("query", query)):
            if [{k: r[k] for k in asdict(samples[0])} for r in manifest[key]] != [asdict(s) for s in samples]:
                raise ValueError("Canonical manifest differs")
        identity = json.loads((source / "identity.json").read_text(encoding="utf-8"))
        if identity["full_protocol_sha256"] != digest(legacy.PROTOCOL_PATH) or identity["manifest_sha256"] != digest(source / "manifest.json"):
            raise ValueError("Parent identity differs")
        identity.update(version="joint_v1", full_protocol_sha256=digest(PROTOCOL_PATH))
        for name, value in (("manifest.json", manifest), ("identity.json", identity)):
            path = folder / name
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != value:
                    raise ValueError("Existing joint identity changed")
            else:
                if name == "manifest.json":
                    path.write_bytes((source / name).read_bytes())
                else:
                    write_json(path, value)
        # Exact manifest bytes are required by identity, so recheck after writing.
        if digest(folder / "manifest.json") != identity["manifest_sha256"]:
            raise ValueError("Copied manifest bytes differ")
        budget = json.loads((source / "image_budget.json").read_text(encoding="utf-8"))
        budget["protocol_sha256"] = digest(PROTOCOL_PATH)
        path = folder / "image_budget.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != budget:
            raise ValueError("Existing image budget changed")
        write_json(path, budget)
    print("CPU joint manifests prepared; no model/CUDA calls", flush=True)


def stages():
    for dataset in DATASETS:
        base = ("--dataset", dataset)
        yield dataset, "joint_source", cmd("run_internvl35_joint_sources_v1.py", *base)
        for stage in ("train_proj", "train_learn", "train_detriever"):
            yield dataset, stage, cmd("run_internvl35_joint_methods_v1.py", *base, "--stage", stage)
        yield dataset, "clip_source", cmd("run_internvl35_joint_sources_v1.py", *base, "--stage", "clip")
        for method in METHODS:
            yield dataset, f"retrieve_{method}", cmd("run_internvl35_joint_methods_v1.py", *base, "--stage", "retrieve", "--method", method)
        for method in METHODS:
            yield dataset, f"icl_{method}", cmd("run_internvl35_joint_icl_v1.py", *base, "--method", method, "--batch-size", "1")
        yield dataset, "cpu_audit", cmd("audit_internvl35_joint_v1.py", "--allow-incomplete")
    yield "all", "full_audit", cmd("audit_internvl35_joint_v1.py")


def execute(argv, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{now()} {argv!r}\n")
        handle.flush()
        subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def boot_identity():
    result = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def wait_aircraft():
    receipt_path = ANALYSIS / "transition.json"
    current_boot = boot_identity()
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["boot_identity"] != current_boot:
            raise RuntimeError("Windows reboot since transition request; do not automatically recover")
        if receipt.get("state") in ("failed", "complete"):
            raise RuntimeError("Transition already ended; no automatic relaunch")
    else:
        receipt = {"state": "waiting_aircraft", "boot_identity": current_boot,
                   "started_utc": now(), "pid": os.getpid(), "protocol_sha256": digest(PROTOCOL_PATH)}
    if receipt["protocol_sha256"] != digest(PROTOCOL_PATH):
        raise ValueError("Transition binding changed")
    while True:
        status = json.loads((legacy.OUTPUT / "pipeline_logs/status.json").read_text(encoding="utf-8"))
        if status["protocol_sha256"] != digest(legacy.PROTOCOL_PATH):
            raise ValueError("Aircraft parent binding changed")
        if status["state"] == "failed":
            raise RuntimeError("Aircraft failed; do not automatically restore GPU")
        receipt.update(checked_utc=now(), parent_state=status["state"], parent_current=status.get("current"))
        write_json(receipt_path, receipt)
        if status["state"] == "paused":
            for method in METHODS:
                if not (legacy.OUTPUT / "aircraft/predictions" / f"{method}.summary.json").exists():
                    raise RuntimeError("Aircraft paused before all six methods finished")
            if boot_identity() != current_boot:
                raise RuntimeError("Windows boot changed; no automatic GPU restart")
            # The old supervisor may still be returning from its final atomic status write.
            time.sleep(2)
            return receipt
        if status["state"] != "running" or status["current"]["dataset"] != "aircraft":
            raise RuntimeError("Unexpected parent stage; refuse handover")
        time.sleep(15)


def pipeline(resume):
    require_gate()
    path = LOG_ROOT / "status.json"
    planned = list(stages())
    binding = digest(PROTOCOL_PATH)
    if path.exists():
        if not resume:
            raise RuntimeError("Pipeline exists; explicit --resume required")
        status = json.loads(path.read_text(encoding="utf-8"))
        if status["protocol_sha256"] != binding or status["state"] == "complete":
            raise ValueError("Invalid resume identity")
        if status["boot_identity"] != boot_identity():
            raise RuntimeError("Windows reboot: do not automatically recover")
        done = [r for r in status["stages"] if r["state"] == "complete"]
        for r, (dataset, stage, argv) in zip(done, planned):
            if (r["dataset"], r["stage"], r["command"], r["return_code"]) != (dataset, stage, argv, 0):
                raise ValueError("Completed stage plan changed")
        status.setdefault("history", []).extend(r for r in status["stages"] if r["state"] != "complete")
        status["stages"] = done
    else:
        status = {"version": "joint_v1", "state": "running", "protocol_sha256": binding,
                  "boot_identity": boot_identity(), "started_utc": now(), "stages": []}
    status.update(state="running", supervisor_pid=os.getpid(), planned_stages=len(planned))
    write_json(path, status)
    for dataset, stage, argv in planned[len(status["stages"]):]:
        if (LOG_ROOT / "STOP_REQUESTED").exists():
            status.update(state="paused", checked_utc=now())
            write_json(path, status)
            return
        record = {"dataset": dataset, "stage": stage, "command": argv, "state": "running", "started_utc": now()}
        status["stages"].append(record)
        status["current"] = {"dataset": dataset, "stage": stage}
        write_json(path, status)
        log = LOG_ROOT / f"{dataset}_{stage}.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{now()} {argv!r}\n")
            handle.flush()
            worker = subprocess.Popen(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            record["worker_pid"] = worker.pid
            write_json(path, status)
            code = worker.wait()
        record.update(state="complete" if code == 0 else "failed", return_code=code, finished_utc=now(), log=str(log))
        status["checked_utc"] = now()
        if code:
            status.update(state="failed", finished_utc=now())
        write_json(path, status)
        if code:
            raise SystemExit(code)
    status.update(state="complete", finished_utc=now())
    status.pop("current", None)
    write_json(path, status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "handover", "pipeline", "plan"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return
    if args.stage == "plan":
        print(json.dumps(list(stages()), indent=2))
        return
    ensure_protocol()
    if args.stage == "handover":
        transition_lock = lock_transition()
        receipt = None
        try:
            receipt = wait_aircraft()
            with lock_gpu():  # Same OS lock as v4, held continuously across gate and pipeline.
                gpu_free()
                execute(cmd("audit_internvl35_qwen_config_v4.py", "--allow-incomplete"), ANALYSIS / "aircraft_audit.log")
                audit = json.loads((legacy.ANALYSIS / "progress_audit.json").read_text(encoding="utf-8"))
                if any(audit["datasets"]["aircraft"]["methods"][m]["state"] != "complete" for m in METHODS):
                    raise RuntimeError("Aircraft independent audit incomplete")
                receipt.update(state="compatibility", checked_utc=now())
                write_json(ANALYSIS / "transition.json", receipt)
                execute(cmd("check_internvl35_joint_v1.py"), ANALYSIS / "compatibility.log")
                receipt.update(state="running_joint_pipeline", checked_utc=now())
                write_json(ANALYSIS / "transition.json", receipt)
                gpu_free()
                pipeline(False)
            receipt.update(state="complete", finished_utc=now())
            write_json(ANALYSIS / "transition.json", receipt)
        except BaseException as error:
            if receipt is None and (ANALYSIS / "transition.json").exists():
                receipt = json.loads((ANALYSIS / "transition.json").read_text(encoding="utf-8"))
            receipt = receipt or {"boot_identity": boot_identity()}
            receipt.update(state="failed", error=repr(error), finished_utc=now())
            write_json(ANALYSIS / "transition.json", receipt)
            raise
        finally:
            transition_lock.close()
    else:
        with lock_gpu():
            gpu_free()
            pipeline(args.resume)

if __name__ == "__main__":
    main()
