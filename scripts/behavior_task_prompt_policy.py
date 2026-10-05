"""Independent task/generic scientific integrity policy. CPU-safe imports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from collect_behavior_priority_zero import atomic, canonical, digest
from behavior_p2_matched_policy import native_identity

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dtd", "cub")
ARMS = ("task", "generic")
SCIENCE = "behavior_task_conditioning_raw_v1"
OPERATION = "behavior_task_conditioning_raw_cloud_v1"
PLAN = ROOT / "analysis/task_conditioning_prompt_plan_v1/protocol_lock.json"
BINDING = ROOT / "analysis/task_conditioning_prompt_implementation_v1/implementation_lock.json"
PROTOCOL_HASH = "c4620bfd0984879ae5ef318411ed88a83a6ec2a8b3271173ef648b8e56a6dbb8"
COUNTS = {"dtd": {"bank": 3760, "query": 1880}, "cub": {"bank": 5994, "query": 5794}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read(path)) != canonical(value):
            raise ValueError("Immutable task-prompt identity changed: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic(path, value)


def validate_plan(plan):
    body = {k: v for k, v in plan.items() if k != "protocol_hash"}
    if plan.get("protocol_hash") != PROTOCOL_HASH or canonical(body) != PROTOCOL_HASH:
        raise ValueError("Original task/generic protocol identity changed")
    if list(plan["datasets"]) != list(DATASETS) or plan["counts"] != COUNTS:
        raise ValueError("Wrong complete official task/generic allocation")
    if (plan["protocol"]["arms"] != list(ARMS) or plan["protocol"]["generic_prompt"] != "Describe the image briefly."
            or plan["protocol"]["training"] != "none; no W, learned layer weights, width grid or dev-ICL search"):
        raise ValueError("Frozen component contrast changed")
    for rel, expected in plan["code_sha256"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Frozen original prompt/downstream source changed: " + rel)
    for dataset in DATASETS:
        data = plan["datasets"][dataset]
        manifest, labels = data["manifest"], data["class_labels"]
        if labels != sorted({r["label"] for r in manifest["bank"]}) or len({v.casefold() for v in labels}) != len(labels):
            raise ValueError("Wrong legal class inventory")
        bank, query = manifest["bank"], manifest["query"]
        ids = [r["sample_id"] for r in bank + query]
        if len(ids) != len(set(ids)) or any(len(manifest[s]) != COUNTS[dataset][s] for s in ("bank", "query")):
            raise ValueError("Duplicate/incomplete official image split")
        for split in ("bank", "query"):
            for row in manifest[split]:
                path = (ROOT / row["path"]).resolve()
                if row["split"] != split or not path.is_relative_to(ROOT / "datasets") or len(row["image_sha256"]) != 64:
                    raise ValueError("Wrong image path/split/SHA inventory")
    return plan


def validate_binding(plan, binding):
    body = {k: v for k, v in binding.items() if k != "implementation_hash"}
    if (canonical(body) != binding.get("implementation_hash") or binding.get("protocol_hash") != PROTOCOL_HASH
            or binding.get("protocol_file_sha256") != digest(PLAN) or binding.get("scientific_root") != SCIENCE
            or binding.get("operational_root") != OPERATION):
        raise ValueError("Independent implementation binding changed")
    for rel, expected in binding["code_sha256"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Independent scientific code changed: " + rel)
    expected_keys = {f"{d}/{arm}/{phase}" for d in DATASETS for arm in ARMS for phase in ("capture", "generate")}
    if set(binding["job_hashes"]) != expected_keys:
        raise ValueError("Need all eight independent capture/generation job locks")
    for key in expected_keys:
        dataset, arm, phase = key.split("/")
        if binding["job_hashes"][key] != canonical(job_spec(plan, binding, dataset, arm, phase)):
            raise ValueError("Scientific job hash mismatch")
    return binding


def job_spec(plan, binding, dataset, arm, phase):
    if dataset not in DATASETS or arm not in ARMS or phase not in ("capture", "generate"):
        raise ValueError("Unknown prespecified scientific tuple")
    return {"experiment": SCIENCE, "protocol_hash": PROTOCOL_HASH,
            "dataset": dataset, "arm": arm, "phase": phase,
            "manifest_hash": canonical(plan["datasets"][dataset]),
            "code_sha256": binding["code_sha256"], "model_inventory_hash": canonical(plan["model_inventory"]),
            "environment": plan["expected_environment"], "source_dtype": "float16", "shape_per_image": [36, 2560],
            "retrieval": plan["protocol"]["retrieval"], "downstream": plan["protocol"]["downstream"]}


def validate_backups(auth, ready, heartbeat, at=None):
    at = at or datetime.now(timezone.utc)
    for record in (ready, heartbeat):
        for key in ("experiment", "protocol_hash", "implementation_hash", "pod_id", "launch_id"):
            if not auth.get(key) or record.get(key) != auth[key]:
                raise ValueError("Backup is absent or belongs to another scientific execution")
        if (record.get("status") != "actual_backup_succeeded" or not record.get("watcher_identity_verified")
                or not re.fullmatch("[0-9a-f]{64}", record.get("archive_sha256", ""))
                or type(record.get("file_count")) is not int or record["file_count"] <= 0):
            raise ValueError("A ready handshake is not actual independent successful backup")
    stamp = datetime.fromisoformat(heartbeat["utc"])
    if stamp.tzinfo is None or not -15 <= (at - stamp).total_seconds() <= 300:
        raise ValueError("Actual backup heartbeat stale/future; never run paid unprotected")


def gpu_guard(plan, binding):
    if sys.platform != "linux" or ROOT.as_posix() != "/workspace/BehaviorICL" or Path.cwd() != ROOT:
        raise ValueError("Exact cloud Linux workspace only; no local GPU")
    folder = ROOT / "results" / OPERATION
    auth = read(folder / "execution_authorization.json")
    if (auth.get("experiment") != SCIENCE or auth.get("protocol_hash") != PROTOCOL_HASH
            or auth.get("implementation_hash") != binding["implementation_hash"]
            or auth.get("code_sha256") != binding["code_sha256"] or auth.get("cwd") != str(ROOT)
            or not auth.get("pod_id") or auth.get("actual_api_pod_id") != auth["pod_id"]):
        raise ValueError("Wrong independent exact-Pod authorization")
    proof_path = folder / "native_identity_receipt.json"
    if digest(proof_path) != auth.get("native_identity_receipt_sha256"):
        raise ValueError("Native/API evidence is not bound to independent authorization")
    proof = read(proof_path)
    if (proof.get("actual_api_pod_id") != auth["pod_id"]
            or not any(proof.get(k) == auth["pod_id"] for k in ("proc1_pod_id", "export_file_pod_id"))
            or any(proof.get(k) not in (None, auth["pod_id"]) for k in ("proc1_pod_id", "export_file_pod_id", "ssh_process_pod_id"))
            or (proof.get("export_file_pod_id") is not None and not proof.get("export_root_owned_nonwritable_regular"))):
        raise ValueError("Native Pod fields absent/conflicting/untrusted")
    validate_backups(auth, read(folder / "backup_ready.json"), read(folder / "backup_heartbeat.json"))
    native_identity(auth["pod_id"])
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError("Another GPU computation exists")
    import torch
    import importlib.metadata
    for name, expected in plan["expected_environment"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Wrong pinned actual environment: " + name)
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_name(0) != auth.get("device")
            or not any(v in auth["device"] for v in ("4090", "5090", "RTX PRO 4500"))):
        raise ValueError("Wrong exact authorized single GPU")
    cache = ROOT / ".hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / plan["model_inventory"]["revision"]
    for name, expected in plan["model_inventory"]["files_SHA256"].items():
        if digest(cache / name) != expected:
            raise ValueError("Full model/processor bytes changed: " + name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return auth


def top4_numpy(scores):
    """CPU oracle for locked actual-score ordering/ties; no target labels."""
    import numpy as np
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] < 4 or not np.isfinite(scores).all():
        raise ValueError("Need finite complete candidate scores")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :4][:, ::-1]
    return indices, np.take_along_axis(scores, indices, axis=1)


def raw_scores_numpy(query, bank):
    import numpy as np
    query, bank = np.asarray(query, dtype=np.float32), np.asarray(bank, dtype=np.float32)
    if (query.ndim != 3 or bank.ndim != 3 or query.shape[1:] != bank.shape[1:]
            or not np.isfinite(query).all() or not np.isfinite(bank).all()):
        raise ValueError("Wrong paired source shape or nonfinite states")
    query = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), np.float32(1e-12))
    bank = bank / np.maximum(np.linalg.norm(bank, axis=-1, keepdims=True), np.float32(1e-12))
    return np.einsum("qld,bld->qb", query, bank, optimize=True) / query.shape[1]


def validate_selection(selection, bank):
    ids, scores = selection["demo_ids"], selection["scores"]
    by_id = {r["sample_id"]: r for r in bank}
    if (len(ids) != 4 or len(set(ids)) != 4 or not set(ids) <= set(by_id)
            or len(scores) != 4 or not all(math.isfinite(v) for v in scores) or scores != sorted(scores)
            or selection["demo_labels"] != [by_id[i]["label"] for i in ids]):
        raise ValueError("Wrong actual top4 score/order/bank label identity")


def valid_tail(ids, label, codec):
    if (not isinstance(ids, list) or any(type(i) is not int for i in ids) or not ids
            or ids[-1] != codec["eos_token_id"] or codec["eos_token_id"] in ids[:-1]
            or ids[:-1] not in codec["label_tokens"][label]):
        raise ValueError("Output tokens are not the exact constrained legal label plus EOS")


def strict_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError("Truncated JSONL; preserve, never repair")
            result.append(json.loads(line))
    return result


def prediction_prefix(path, queries, selections, context, codec, complete=False):
    rows = strict_jsonl(path)
    for i, row in enumerate(rows):
        if i >= len(queries):
            raise ValueError("Too many official predictions")
        query, selected = queries[i], selections[queries[i]["sample_id"]]
        if (row.get("query_id") != query["sample_id"] or row.get("target") != query["label"]
                or row.get("prediction") not in codec["label_tokens"] or type(row.get("correct")) is not bool
                or row["correct"] != (row["prediction"] == query["label"])
                or row.get("raw_answer", "").strip().casefold() != row["prediction"].casefold()
                or row.get("context_hash") != context["context_hash"] or row.get("job_hash") != context["job_hash"]
                or any(row.get(k) != selected[k] for k in ("demo_ids", "demo_labels", "scores"))
                or type(row.get("input_tokens")) is not int or row["input_tokens"] <= 0
                or row.get("output_tokens") != len(row.get("output_token_ids", []))
                or type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] <= 0
                or len(row.get("prompt_fingerprint", "")) != 64):
            raise ValueError("Wrong official prediction/source/order/token/time identity")
        valid_tail(row["output_token_ids"], row["prediction"], codec)
    if complete and len(rows) != len(queries):
        raise ValueError("Incomplete official test; no partial accuracy")
    return rows


def finite_array(array):
    import numpy as np
    return all(np.isfinite(array[i:i + 64]).all() for i in range(0, len(array), 64))


def capture_prefix(logs, progress, rows, identity, array):
    count = progress["completed"]
    if (type(count) is not int or not 0 <= count <= len(rows) or len(logs) != count
            or any(progress.get(k) != identity[k] for k in ("job_hash", "shape", "dtype"))
            or list(array.shape) != identity["shape"] or str(array.dtype) != identity["dtype"]):
        raise ValueError("Source array/log/committed prefix disagreement")
    for i, entry in enumerate(logs):
        expected = rows[i]
        if (entry.get("sample_id") != expected["sample_id"] or entry.get("image_sha256") != expected["image_sha256"]
                or entry.get("job_hash") != identity["job_hash"] or entry.get("output_tokens") != 0
                or entry.get("input_tokens", 0) <= 0 or entry.get("last_input_position") != entry["input_tokens"] - 1
                or len(entry.get("prompt_fingerprint", "")) != 64
                or type(entry.get("seconds")) not in (int, float) or not math.isfinite(entry["seconds"]) or entry["seconds"] <= 0):
            raise ValueError("Wrong actual source ID/image/prompt/token/time log")
    if not finite_array(array[:count]):
        raise ValueError("Nonfinite committed hidden-state prefix")
    return count
