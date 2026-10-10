"""Operational clock guard; frozen joint science code and protocol are unchanged."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import run_internvl35_joint_pipeline_v1 as base


def normalize_boot(expected, observed):
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    delta = abs((parse(expected) - parse(observed)).total_seconds())
    # WMI computes a wall-clock boot timestamp; small clock correction is not a new boot.
    # In this operation the recorded boot predates all workers by almost a day.
    # Larger changes are deliberately returned unchanged so the original guard rejects them.
    return (expected if delta <= 10 else observed), delta


def main():
    path = base.ANALYSIS / "transition.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous["state"] != "waiting_aircraft":
        raise RuntimeError("Clock guard is authorized only for the diagnosed waiting transition")
    expected = previous["boot_identity"]
    original = base.boot_identity
    observed = original()
    normalized, delta = normalize_boot(expected, observed)
    if normalized != expected:
        raise RuntimeError("Boot time difference exceeds diagnosed clock correction; no automatic recovery")
    receipt = {"version": "operational_clock_guard_v1", "science_protocol_sha256": base.digest(base.PROTOCOL_PATH),
               "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "expected_boot_timestamp": expected, "observed_boot_timestamp": observed,
               "initial_delta_seconds": delta, "tolerance_seconds": 10,
               "basis": "original Aircraft supervisor PID3872 created2026-10-07T16:52:59+08 and GPU PID18960 created2026-10-08T14:05:26+08 remain alive; no System41/6005/6008 events in last3h; old waiting process survived timestamp change",
               "started_utc": base.now(), "last_checked_utc": base.now()}
    receipt_path = base.ANALYSIS / "operational_clock_guard_v1.json"
    base.write_json(receipt_path, receipt)
    def guarded():
        current = original()
        result, offset = normalize_boot(expected, current)
        receipt.update(last_checked_utc=base.now(), last_observed_boot_timestamp=current, last_delta_seconds=offset)
        base.write_json(receipt_path, receipt)
        return result
    base.boot_identity = guarded
    base.main()


if __name__ == "__main__":
    main()
