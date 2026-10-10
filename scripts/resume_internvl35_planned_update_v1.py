"""Reviewed recovery from confirmed planned Windows upgrade, not crash recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import internvl35_joint_config_v2 as config
import run_internvl35_joint_pipeline_v2 as base

def main():
    config.require_gate()
    evidence_path = config.ANALYSIS / "windows_update_restart_20261009.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    if not any(e["id"]==1074 and "TrustedInstaller.exe" in e["message"] and "0x80020003" in e["message"]
               for e in evidence["system_events"]):
        raise RuntimeError("No confirmed planned Windows upgrade event")
    if any(e["id"] in (41,1001,6008) for e in evidence["system_events"]):
        raise RuntimeError("Unexpected crash event; do not automatically recover")
    path = base.LOG_ROOT / "status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    if status["current"] != {"dataset":"cub","stage":"icl_raw"}:
        raise RuntimeError("Wrong interrupted stage")
    boot = base.boot_identity()
    if boot != evidence["current_boot"]:
        raise RuntimeError("System boot changed after diagnosis")
    receipt_path = config.ANALYSIS / "planned_update_resume_20261009_v1.json"
    with base.lock_transition(),base.lock_gpu():
        base.gpu_free()
        if receipt_path.exists():
            raise RuntimeError("Planned update operation already consumed; no retry loop")
        snapshot = config.ANALYSIS / "status.pre_planned_update_resume_20261009_v1.json"
        snapshot.write_bytes(path.read_bytes())
        receipt = {"state":"compatibility_replay", "started_utc":config.now(),"pid":os.getpid(),
            "authorization":"Existing human continuation remains active; confirmed planned update interruption, no crash",
            "previous_boot_identity":status["boot_identity"],"boot_identity":boot,
            "protocol_sha256":config.digest(config.PROTOCOL_PATH),
            "evidence_sha256":config.digest(evidence_path),
            "code_sha256":config.digest(Path(__file__)),
            "replay_code_sha256":config.digest(config.ROOT/"scripts/check_internvl35_restart_replay_v1.py"),
            "saved_raw_predictions":2688}
        config.write_json(receipt_path,receipt)
        try:
            report = config.ANALYSIS / "planned_update_replay_20261009_v1.json"
            base.execute(base.cmd("check_internvl35_restart_replay_v1.py","--report",str(report)),
                         config.ANALYSIS/"planned_update_replay_20261009_v1.log")
            replay = json.loads(report.read_text(encoding="utf-8"))
            if replay["state"]!="complete" or not replay["exact_saved_output"]:
                raise RuntimeError("Saved-context replay failed")
            base.gpu_free()
            # Boot is mutable operational metadata; save the original bytes first.
            # Scientific model/source/selection/prediction identities remain untouched.
            status.update(state="paused",boot_identity=boot,resume_reason="Confirmed planned Windows upgrade; saved-context replay exact")
            config.write_json(path,status)
            receipt.update(state="running",replay_sha256=config.digest(report),checked_utc=config.now())
            config.write_json(receipt_path,receipt)
            transition_path = config.ANALYSIS/"transition.json"
            transition = json.loads(transition_path.read_text(encoding="utf-8"))
            transition.update(state="running_joint_pipeline",boot_identity=boot,operation_receipt=str(receipt_path),checked_utc=config.now())
            config.write_json(transition_path,transition)
            base.pipeline(True)
            receipt.update(state="complete",finished_utc=config.now())
            config.write_json(receipt_path,receipt)
        except BaseException as error:
            receipt.update(state="failed",error=repr(error),finished_utc=config.now())
            config.write_json(receipt_path,receipt)
            raise

if __name__=="__main__":
    main()
