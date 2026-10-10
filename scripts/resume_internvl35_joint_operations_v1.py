"""Resume frozen joint_v2; account for observed Windows desktop GPU contexts."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import internvl35_joint_config_v2 as config
import run_internvl35_joint_pipeline_v2 as base


def gpu_science_free():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                            text=True, capture_output=True, check=True)
    contexts = list(csv.reader(result.stdout.splitlines()))
    if not contexts:
        return []
    code = "@(Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath,@{Name='CreatedUtc';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}}) | ConvertTo-Json -Compress"
    process_result = subprocess.run(["powershell", "-NoProfile", "-Command", code],
                                    text=True, capture_output=True, check=True)
    processes = json.loads(process_result.stdout)
    if isinstance(processes, dict):
        processes = [processes]
    by_pid = {int(p["ProcessId"]): p for p in processes}
    observed_desktop = []
    for row in contexts:
        pid = int(row[0].strip())
        info = by_pid.get(pid)
        if info is None:
            raise RuntimeError(f"GPU context PID{pid} cannot be identified")
        name = info["Name"].lower()
        path = (info.get("ExecutablePath") or "").lower().replace("/", "\\")
        if name == "dwm.exe" and pid == 8904:
            # Read-only CIM attestation for this boot's protected desktop process.
            # Its path is hidden by Windows; never allow an arbitrary name-only match.
            if info.get("CreatedUtc") != "2026-10-08T09:25:59.9376480Z":
                raise RuntimeError("Protected DWM creation identity changed")
        elif name == "shellexperiencehost.exe" and path == "c:\\windows\\systemapps\\shellexperiencehost_cw5n1h2txyewy\\shellexperiencehost.exe":
            pass
        else:
            raise RuntimeError(f"GPU has an unapproved compute context: PID{pid}, {info['Name']}")
        observed_desktop.append(info)
    return observed_desktop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config.require_gate()
    status_path = base.LOG_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["state"] != "paused" or status["boot_identity"] != base.boot_identity():
        raise RuntimeError("Explicit resume requires a paused same-boot pipeline")
    if args.check:
        print(json.dumps({"state": "CPU_checked", "desktop_contexts": gpu_science_free()}, ensure_ascii=True))
        return
    with base.lock_transition(), base.lock_gpu():
        desktop = gpu_science_free()
        stamp = config.now().replace(":", "").replace(".", "_")
        receipt_path = config.ANALYSIS / f"explicit_resume_{stamp}.json"
        snapshot = config.ANALYSIS / f"status.pre_resume_{stamp}.json"
        snapshot.write_bytes(status_path.read_bytes())
        receipt = {"state": "running", "started_utc": config.now(), "authorization": "Human explicitly requested continue after18:58 pause",
                   "protocol_sha256": config.digest(config.PROTOCOL_PATH), "code_sha256": config.digest(Path(__file__)),
                   "boot_identity": status["boot_identity"], "desktop_contexts": desktop,
                   "science_gpu_workers": 0, "saved_cub_rows": 2603, "status_snapshot": str(snapshot)}
        config.write_json(receipt_path, receipt)
        flag = base.LOG_ROOT / "STOP_REQUESTED"
        if flag.exists():
            flag.rename(flag.with_name(f"STOP_REQUESTED.user_pause_{stamp}"))
        transition_path = config.ANALYSIS / "transition.json"
        transition = json.loads(transition_path.read_text(encoding="utf-8"))
        transition.update(state="running_joint_pipeline", resumed_utc=config.now(), operation_receipt=str(receipt_path))
        config.write_json(transition_path, transition)
        try:
            base.pipeline(True)
            receipt.update(state="complete", finished_utc=config.now())
            config.write_json(receipt_path, receipt)
            transition.update(state="complete", finished_utc=config.now())
            config.write_json(transition_path, transition)
        except BaseException as error:
            receipt.update(state="failed", error=repr(error), finished_utc=config.now())
            config.write_json(receipt_path, receipt)
            raise

if __name__ == "__main__":
    main()
