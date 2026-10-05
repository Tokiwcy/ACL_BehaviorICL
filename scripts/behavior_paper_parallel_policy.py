"""CPU-only policy for the newly authorized extra independently monitored card.

No launch, CUDA, checkpoint or scientific mutation. An active-job registry is
required before new parallel entries are permitted; old serial entries remain
immutable. Fees add independent segments, never another segment's live total.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import math

import behavior_task_prompt_cloud as prompt

ENDED_BASELINE = 6.1720589067
MAX_CARDS = 2
PAIRED_ROLES = {"task_prompt", "full_labels"}


def authority(request):
    if (request.get("autonomous_cloud_capacity_authorized") is not True
            or request.get("parallel_ablation_capacity_authorized") is not True
            or request.get("parallel_ablation_max_cloud_GPUs_authorized") != MAX_CARDS
            or request.get("parallel_ablation_human_quote") != "如果你觉得可以的话，多个ablation可以多搞一张卡去跑。都做好监控就行"
            or request.get("local_gpu_execution_authorized") is not False):
        raise ValueError("Missing direct independent-extra-card authority")


def validate_registry(rows, root):
    root = Path(root).resolve()
    if not isinstance(rows, list) or len(rows) > MAX_CARDS:
        raise ValueError("At most two registered active scientific jobs")
    pods, launches, roles, backups, sciences = set(), set(), set(), set(), set()
    for row in rows:
        role = row.get("role")
        if role not in PAIRED_ROLES or role in roles:
            raise ValueError("Only distinct agreed prompt and full-label tasks may overlap")
        if row.get("pod_id") in pods or row.get("launch_id") in launches or not row.get("pod_id") or not row.get("launch_id"):
            raise ValueError("Independent exact Pod and launch required")
        folder = (root / row["backup_dir"]).resolve()
        science = (root / "results" / row["scientific_root"]).resolve()
        if (not folder.is_relative_to(root / "results/cloud_trial") or folder.name != "session_1"
                or not science.is_relative_to(root / "results") or science == root / "results"
                or folder in backups or science in sciences):
            raise ValueError("Unsafe or shared writable science/backup directory")
        if row.get("single_GPU_count") != 1 or row.get("actual_backup_required") is not True:
            raise ValueError("Each card must have sole computation and actual independent backup")
        roles.add(role); pods.add(row["pod_id"]); launches.add(row["launch_id"])
        backups.add(folder); sciences.add(science)
    return rows


def prompt_smoke_gate(smoke, binding, plan, environment, actual_backup):
    """Compatibility only, not complete official test or new accuracy evidence."""
    dataset = "dtd"
    data = plan["datasets"][dataset]
    if (smoke.get("status") != "actual_two_prompt_sources_and_four_demo_legal_generation_passed"
            or smoke.get("implementation_hash") != binding["implementation_hash"]
            or smoke.get("arms") != ["task", "generic"]
            or smoke.get("diagnostic_demo_ids") != [r["sample_id"] for r in data["manifest"]["bank"][:4]]
            or smoke.get("query_id") != data["manifest"]["query"][0]["sample_id"]
            or smoke.get("counted_as_test_prediction") is not False):
        raise ValueError("Actual fixed-context first-dataset source/legal smoke not proven")
    if (environment.get("status") != "CPU_pinned_runtime_full_model_SHA_verified_not_generation"
            or environment.get("versions") != plan["expected_environment"]
            or environment.get("model_inventory") != plan["model_inventory"]):
        raise ValueError("Complete exact model/runtime audit missing")
    if (actual_backup.get("status") != "actual_backup_succeeded"
            or actual_backup.get("watcher_identity_verified") is not True
            or not actual_backup.get("archive_sha256") or not actual_backup.get("file_count")):
        raise ValueError("Real independently retrieved backup required")
    return {"compatibility_gate_passed": True, "complete_official_test": False,
            "independent_second_card_has_own_smokes": True}


def cumulative(segments, at=None):
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        raise ValueError("Aware accounting timestamp required")
    result, identities = [], set()
    for account in segments:
        identity = (account.get("pod_id"), account.get("launch_id"))
        if not all(identity) or identity in identities:
            raise ValueError("Missing/duplicate exact fee segment")
        identities.add(identity)
        start = datetime.fromisoformat(account["start_request_utc"])
        end = datetime.fromisoformat(account["confirmed_utc"]) if account.get("confirmed_utc") else at
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError("Invalid actual start/earliest-stop segment")
        rate = account["cost_per_hour"]
        if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate < 0:
            raise ValueError("Invalid verified live hourly rate")
        seconds = 0 if account.get("explicit_capacity_rejection") is True else (end - start).total_seconds()
        fee = seconds * rate / 3600
        result.append({"pod_id": identity[0], "launch_id": identity[1], "seconds": seconds,
                       "GPU_estimate_USD": fee, "ended": bool(account.get("confirmed_utc"))})
    return {"ended_prior_baseline_USD": ENDED_BASELINE, "segments": result,
            "cumulative_GPU_estimate_USD": ENDED_BASELINE + sum(r["GPU_estimate_USD"] for r in result),
            "estimate_not_invoice": True, "storage_included": False,
            "stopped_gaps_or_unknown_account_charges_included": False}
