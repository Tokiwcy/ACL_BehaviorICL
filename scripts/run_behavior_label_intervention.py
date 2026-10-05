"""Independent fixed-context label intervention; default is CPU preflight.

Capture is cloud-only and requires a new exact-Pod execution authorization and
fresh successful backup handshake. This runner cannot resume P0 or old queues.
It changes only the prescribed demo labels and uses the original model adapter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from attach_behavior_label_inputs import ARMS, EXPECTED_PLAN_HASH, validate_plan
from collect_behavior_priority_zero import atomic, canonical, digest, lock, MODEL, REVISION

DATASETS = ("dtd", "aircraft", "cub", "pets", "dogs")
SCIENCE_GROUP = "behavior_corrective_labels_v1"
OPERATION_GROUP = "behavior_corrective_labels_cloud_v1"
CODE = ("scripts/run_behavior_label_intervention.py", "scripts/attach_behavior_label_inputs.py",
        "scripts/prepare_behavior_label_intervention.py", "scripts/collect_behavior_priority_zero.py",
        "scripts/multimodal_model_adapter.py", "scripts/legal_label_decoding.py",
        "scripts/multidataset_protocol.py", "scripts/run_behavior_priority_analysis.py",
        "scripts/analyze_behavior_paper_p0.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_inputs(inputs, base):
    validate_plan(base)
    if (inputs.get("base_plan_hash") != EXPECTED_PLAN_HASH or inputs.get("model") != MODEL
            or inputs.get("revision") != REVISION or inputs.get("paired_arms") != list(ARMS)
            or canonical({k: v for k, v in inputs.items() if k != "attachment_hash"}) != inputs.get("attachment_hash")):
        raise ValueError("Wrong attached input identity/model/arms")
    if len(inputs["all_queries"]) != len(base["all_queries"]):
        raise ValueError("Query dropped or added after lock")
    for row, original in zip(inputs["all_queries"], base["all_queries"]):
        for field, value in original.items():
            if field not in ("demo_own_zero_labels", "own_arm_status") and row.get(field) != value:
                raise ValueError("Original locked context changed: " + field)
        if len(row["own_zero_evidence"]) != 4 or len(row["demo_images_in_actual_prompt_order"]) != 4:
            raise ValueError("Incomplete demo evidence/images")
        classes = row["legal_classes"]
        if len(set(v.casefold() for v in classes)) != len(classes):
            raise ValueError("Casefold-colliding legal classes")
        legal = {v.casefold(): v for v in classes}
        preds = []
        for i, gold, image, z in zip(row["demo_ids_in_actual_prompt_order"], row["gold_labels"],
                                     row["demo_images_in_actual_prompt_order"], row["own_zero_evidence"]):
            pred = legal.get(z["raw_zero_shot_answer"].strip().casefold())
            if (z["demo_id"] != i or z["gold_label"] != gold or image["sample_id"] != i
                    or z["predicted_label"] != pred or z["valid"] != (pred is not None)):
                raise ValueError("Attached own behavior changed")
            preds.append(pred)
        status = "ready" if all(p is not None for p in preds) else "missing_illegal_demo_zero"
        if (row["demo_own_zero_labels"] != preds or row["own_arm_status"] != status
                or row["own_legal_labels_changed"] != sum(p is not None and p != y for p, y in zip(preds, row["gold_labels"]))
                or row["invalid_demo_zero_count"] != sum(p is None for p in preds)
                or row["hashed_wrong_labels_changed"] != 4
                or row["query_image"]["sample_id"] != row["query_id"]):
            raise ValueError("Changed attachment status/counts/query image")


def schedule(rows):
    # Original ID-hash order, each query paired across arms, no outcome filtering.
    return [(row, arm) for row in rows for arm in ARMS]


def prompt_spec(row, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    labels = {"gold_demo_labels": row["gold_labels"],
              "demo_own_legal_zero_labels": row["demo_own_zero_labels"],
              "hashed_wrong_demo_labels": row["hashed_wrong_labels"]}
    zero = arm == "same_constrained_zero"
    return {"dataset": row["dataset"], "query_id": row["query_id"], "arm": arm,
            "query_image": row["query_image"], "legal_classes": row["legal_classes"],
            "demo_images": [] if zero else row["demo_images_in_actual_prompt_order"],
            "demo_labels": [] if zero else labels[arm],
            "adapter": "zero_shot_messages" if zero else "icl_messages",
            "image_size_hint": 224, "vision_pixels_override": None,
            "decoding": "original greedy legal-label trie, batch1"}


def image_path(record, root=ROOT):
    rel = Path(record["relative_path"])
    path = (root / rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root / "datasets"):
        raise ValueError("Image path escapes official datasets")
    if digest(path) != record["sha256"]:
        raise ValueError("Selected image bytes changed: " + record["sample_id"])
    return path


def verify_images(rows):
    seen = {}
    for row in rows:
        for record in [row["query_image"], *row["demo_images_in_actual_prompt_order"]]:
            rel = record["relative_path"]
            if rel in seen:
                if seen[rel] != record["sha256"]:
                    raise ValueError("Image SHA conflict")
            else:
                image_path(record)
                seen[rel] = record["sha256"]
    return len(seen)


def job_identity(inputs, dataset, cache_dir):
    snapshot = cache_dir / f"models--Qwen--Qwen3-VL-4B-Instruct/snapshots/{REVISION}"
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
             "chat_template.json", "model.safetensors.index.json")
    configs = {name: digest(snapshot / name) for name in files}
    index = read(snapshot / "model.safetensors.index.json")
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        relative = Path(name)
        path = (snapshot / name).resolve()
        # Hugging Face snapshots may link to blobs within the same cache.
        if (relative.is_absolute() or ".." in relative.parts
                or not path.is_relative_to(cache_dir.resolve()) or path.stat().st_size < 1000):
            raise ValueError("Missing/unsafe fixed snapshot weights")
        shards[name] = path.stat().st_size
    body = {"experiment": SCIENCE_GROUP, "dataset": dataset, "model": MODEL, "revision": REVISION,
            "base_plan_hash": EXPECTED_PLAN_HASH, "attachment_hash": inputs["attachment_hash"],
            "rows": [r["query_id"] for r in inputs["all_queries"] if r["dataset"] == dataset],
            "arm_order": list(ARMS), "code_sha256": {name: digest(ROOT / name) for name in CODE},
            "model_config_sha256": configs, "fixed_revision_weight_file_sizes": shards,
            "vision_override": None, "image_size_hint": 224, "batch_size": 1,
            "decoding": "original adapter greedy legal-label trie max longest label tokenization +EOS",
            "own_missing_policy": "explicit missing, no generation or repair; other arms keep same query",
            "gold_reference": "new run, same environment; no historical accuracy substitution"}
    body["job_hash"] = canonical(body)
    return body


def expected_record(row, arm, job_hash):
    request = prompt_spec(row, arm)
    skipped = arm == "demo_own_legal_zero_labels" and row["own_arm_status"] != "ready"
    return {"dataset": row["dataset"], "query_id": row["query_id"], "arm": arm, "job_hash": job_hash,
            "request_sha256": canonical(request), "target_for_evaluation_only": row["target_for_evaluation_only"],
            "demo_ids": [r["sample_id"] for r in request["demo_images"]],
            "demo_labels": request["demo_labels"], "image_sha256": [request["query_image"]["sha256"],
                                                                       *[r["sha256"] for r in request["demo_images"]]],
            "status": "missing_illegal_demo_zero" if skipped else "generated"}


def validate_prefix(path, jobs, job_hash):
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                raise ValueError("Truncated JSONL: preserve failure, never silently repair")
            value = json.loads(line)
            if len(records) >= len(jobs):
                raise ValueError("Extra intervention output")
            row, arm = jobs[len(records)]
            expected = expected_record(row, arm, job_hash)
            if any(value.get(k) != v for k, v in expected.items()):
                raise ValueError("Intervention prefix context/order/job mismatch")
            if expected["status"] != "generated":
                if any(value.get(k) is not None for k in ("raw_answer", "prediction", "correct")):
                    raise ValueError("Missing own arm fabricated a prediction")
                if value.get("output_tokens") != 0 or value.get("elapsed_seconds") != 0:
                    raise ValueError("Missing own arm must not claim GPU generation")
            else:
                raw = value.get("raw_answer")
                if not isinstance(raw, str):
                    raise ValueError("Missing literal generated answer")
                legal = {v.casefold(): v for v in row["legal_classes"]}
                pred = legal.get(raw.strip().casefold())
                if (pred is None or value.get("prediction") != pred or type(value.get("correct")) is not bool
                        or value["correct"] != (pred == row["target_for_evaluation_only"])):
                    raise ValueError("Illegal or relabeled constrained output")
                if (type(value.get("output_tokens")) is not int or value["output_tokens"] <= 0
                        or type(value.get("input_tokens")) is not int or value["input_tokens"] <= 0
                        or type(value.get("elapsed_seconds")) not in (int, float)
                        or not math.isfinite(value["elapsed_seconds"]) or value["elapsed_seconds"] <= 0):
                    raise ValueError("Missing actual token/time evidence")
                if len(value.get("demo_label_token_counts", [])) != len(expected["demo_labels"]):
                    raise ValueError("Missing demo-label length accounting")
                if any(type(n) is not int or n <= 0 for n in value["demo_label_token_counts"]):
                    raise ValueError("Invalid label token accounting")
                tokens = value.get("generated_token_ids")
                if (not isinstance(tokens, list) or len(tokens) != value["output_tokens"]
                        or any(type(t) is not int or t < 0 for t in tokens)
                        or not re.fullmatch("[0-9a-f]{64}", value.get("prompt_messages_sha256", ""))):
                    raise ValueError("Missing actual generated token IDs or prompt hash")
            records.append(value)
    return records


def assert_supervised_capture(authorization, ready, job, expected_pod, now=None):
    now = now or datetime.now(timezone.utc)
    if (authorization.get("status") != "audited_independent_label_entry_authorized"
            or authorization.get("experiment") != OPERATION_GROUP
            or authorization.get("pod_id") != expected_pod
            or authorization.get("attachment_hash") != job["attachment_hash"]
            or authorization.get("base_plan_hash") != EXPECTED_PLAN_HASH
            or authorization.get("job_hashes", {}).get(job["dataset"]) != job["job_hash"]
            or authorization.get("max_concurrent_gpu_jobs") != 1
            or authorization.get("old_queues_authorized") is not False):
        raise RuntimeError("Missing new exact-Pod audited execution authorization")
    if (ready.get("status") != "actual_backup_succeeded" or ready.get("experiment") != OPERATION_GROUP
            or ready.get("pod_id") != expected_pod or ready.get("attachment_hash") != job["attachment_hash"]
            or not authorization.get("launch_id") or ready.get("launch_id") != authorization["launch_id"]):
        raise RuntimeError("No matching independent supervisor backup handshake")
    stamp = datetime.fromisoformat(ready["utc"])
    if stamp.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= 600:
        raise RuntimeError("Supervisor backup readiness is stale or from the future")


def render_messages(row, arm):
    from multimodal_model_adapter import icl_messages, zero_shot_messages
    from multidataset_protocol import Sample
    request = prompt_spec(row, arm)
    # Query truth is deliberately unavailable to the message adapter.
    sample = Sample(row["query_id"], str(image_path(row["query_image"])), "", "query")
    if arm == "same_constrained_zero":
        return zero_shot_messages(sample, row["dataset"], row["legal_classes"], 224)
    if any(label is None for label in request["demo_labels"]):
        raise ValueError("Illegal own labels must skip generation, never render a repaired prompt")
    demos = [Sample(image["sample_id"], str(image_path(image)), label, "bank")
             for image, label in zip(request["demo_images"], request["demo_labels"])]
    return icl_messages(demos, sample, row["dataset"], row["legal_classes"], 224)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("preflight", "capture", "verify"), default="preflight")
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--inputs-lock", type=Path, default=ROOT / "analysis/paired_label_inputs_v1/attached_inputs_lock.json")
    p.add_argument("--base-plan", type=Path, default=ROOT / "analysis/paired_label_plan_v1/query_arm_lock.json")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    p.add_argument("--output-root", type=Path, default=ROOT / "results" / SCIENCE_GROUP)
    p.add_argument("--allow-gpu", action="store_true")
    p.add_argument("--expected-device")
    p.add_argument("--expected-pod-id")
    p.add_argument("--execution-authorization", type=Path)
    p.add_argument("--backup-ready", type=Path)
    p.add_argument("--max-records", type=int, help="Audited smoke prefix only, never a complete result")
    p.add_argument("--preflight-receipt", type=Path, help="Fresh CPU-only identity receipt under analysis/")
    args = p.parse_args()
    inputs, base = read(args.inputs_lock), read(args.base_plan)
    validate_inputs(inputs, base)
    if digest(args.base_plan) != inputs["base_plan_file_sha256"]:
        raise ValueError("Base plan bytes changed")
    rows = [r for r in inputs["all_queries"] if r["dataset"] == args.dataset]
    if len(rows) != 256 or len(set(r["query_id"] for r in rows)) != 256:
        raise ValueError("Incomplete/duplicate fixed per-dataset query cohort")
    image_count = verify_images(rows)
    job = job_identity(inputs, args.dataset, args.cache_dir.resolve())
    jobs = schedule(rows)
    if args.phase == "preflight":
        receipt = {"status": "CPU_preflight_passed_no_GPU", "dataset": args.dataset,
                   "job_hash": job["job_hash"], "queries": len(rows), "planned_records": len(jobs),
                   "own_arm_missing": sum(r["own_arm_status"] != "ready" for r in rows),
                   "selected_image_sha_checked": image_count, "identity": job,
                   "input_lock_file_sha256": digest(args.inputs_lock),
                   "base_plan_file_sha256": digest(args.base_plan),
                   "actual_generation_smoke": "not executed", "GPU_execution": False,
                   "utc": datetime.now(timezone.utc).isoformat()}
        if args.preflight_receipt:
            path = args.preflight_receipt.resolve()
            if not path.is_relative_to(ROOT / "analysis") or path.exists():
                raise ValueError("Preflight receipt must be a fresh derived analysis file")
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "identity"}, ensure_ascii=False))
        return
    dest = args.output_root.resolve() / args.dataset
    if not dest.is_relative_to(ROOT / "results" / SCIENCE_GROUP):
        raise ValueError("Use independent label intervention result group only")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("max-records must be positive")
    if args.phase == "capture":
        if (not args.allow_gpu or not args.expected_device or not args.expected_pod_id
                or not args.execution_authorization or not args.backup_ready
                or sys.platform != "linux" or os.environ.get("RUNPOD_POD_ID") != args.expected_pod_id):
            raise RuntimeError("Cloud-only capture needs explicit exact-Pod GPU/supervisor authorization; local GPU denied")
        def supervised():
            assert_supervised_capture(read(args.execution_authorization), read(args.backup_ready), job, args.expected_pod_id)
        supervised()
        dest.mkdir(parents=True, exist_ok=True)
    elif not dest.exists():
        raise FileNotFoundError("No label intervention has been executed")
    with lock(dest / "capture.lock"):
        ip, outputs = dest / "identity.json", dest / "predictions.jsonl"
        if ip.exists():
            if canonical(read(ip)) != canonical(job):
                raise ValueError("Intervention job identity changed")
        elif args.phase == "verify":
            raise FileNotFoundError(ip)
        else:
            atomic(ip, job)
        prefix = validate_prefix(outputs, jobs, job["job_hash"])
        if args.phase == "verify":
            if len(prefix) != len(jobs):
                raise ValueError("Incomplete intervention; do not report partial accuracy")
            runtime_files = list(dest.glob("runtime_*.json"))
            if not runtime_files:
                raise ValueError("No actual generation runtime receipts")
            for path in runtime_files:
                runtime = read(path)
                if (runtime.get("job_hash") != job["job_hash"] or not runtime.get("pod_id")
                        or not str(runtime.get("torch", "")).startswith("2.8.0")
                        or runtime.get("transformers") != "5.13.0"
                        or type(runtime.get("model_load_seconds")) not in (int, float)
                        or not math.isfinite(runtime["model_load_seconds"]) or runtime["model_load_seconds"] <= 0):
                    raise ValueError("Wrong or missing actual pinned runtime receipt")
            atomic(dest / "verification.json", {"status": "all_complete_verified", "job_hash": job["job_hash"],
                    "records": len(prefix), "queries": len(rows), "outputs_sha256": digest(outputs),
                    "generated": sum(r["status"] == "generated" for r in prefix),
                    "explicit_missing": sum(r["status"] != "generated" for r in prefix),
                    "runtime_sha256": {p.name: digest(p) for p in runtime_files}})
            print("Complete label intervention verified: " + args.dataset)
            return
        start = len(prefix)
        if start == len(jobs):
            print("Already captured; verify without rerunning any arm")
            return
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
            raise RuntimeError("Another GPU compute process present")
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        import transformers
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or torch.cuda.get_device_name(0) != args.expected_device
                or not torch.__version__.startswith("2.8.0") or transformers.__version__ != "5.13.0"):
            raise RuntimeError("Wrong exact GPU or pinned generation environment")
        from multimodal_model_adapter import load_model, MODEL_SPECS, prepare_batch
        from legal_label_decoding import label_token_sequences, LegalLabelTrie, exact_generated_label
        loaded = time.perf_counter()
        model, processor = load_model(MODEL_SPECS["qwen3vl4b"], args.cache_dir.resolve(), None)
        torch.cuda.synchronize()
        atomic(dest / ("runtime_" + str(start) + ".json"), {"device": args.expected_device, "pod_id": args.expected_pod_id,
                "torch": torch.__version__, "transformers": transformers.__version__, "pid": os.getpid(),
                "model_load_seconds": time.perf_counter() - loaded, "job_hash": job["job_hash"],
                "environment_equivalence_with_historical_ICL": "not asserted", "utc": datetime.now(timezone.utc).isoformat()})
        sequences, maximum = label_token_sequences(processor.tokenizer, rows[0]["legal_classes"])
        trie = LegalLabelTrie(sequences, processor.tokenizer.eos_token_id)
        stop = len(jobs) if args.max_records is None else min(len(jobs), start + args.max_records)
        with outputs.open("a", encoding="utf-8") as f:
            for i in range(start, stop):
                supervised()
                row, arm = jobs[i]
                result = expected_record(row, arm, job["job_hash"])
                if result["status"] != "generated":
                    result.update(raw_answer=None, prediction=None, correct=None, output_tokens=0,
                                  input_tokens=0, elapsed_seconds=0, demo_label_token_counts=None)
                else:
                    messages = render_messages(row, arm)
                    torch.cuda.synchronize(); before = time.perf_counter()
                    batch = prepare_batch(processor, [messages]); n = int(batch["input_ids"].shape[1])
                    with torch.inference_mode():
                        generated = model.generate(**batch, max_new_tokens=maximum, do_sample=False, use_cache=True,
                                                    prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(n))
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - before
                    tail = generated[:, n:]
                    raw = processor.batch_decode(tail, skip_special_tokens=True)[0].strip()
                    pred = exact_generated_label(raw, row["legal_classes"])
                    result.update(raw_answer=raw, prediction=pred, correct=pred == row["target_for_evaluation_only"],
                                  output_tokens=int(tail.numel()), input_tokens=n, elapsed_seconds=elapsed,
                                  generated_token_ids=tail[0].tolist(), prompt_messages_sha256=canonical(messages),
                                  demo_label_token_counts=[len(processor.tokenizer.encode(" " + v, add_special_tokens=False))
                                                           for v in result["demo_labels"]])
                    # Source bytes must still match after the real model call.
                    for image in [row["query_image"], *row["demo_images_in_actual_prompt_order"]]:
                        image_path(image)
                    del batch, generated, tail
                f.write(json.dumps(result, ensure_ascii=False) + "\n"); f.flush(); os.fsync(f.fileno())
                atomic(dest / "progress.json", {"status": "capture_complete_pending_verify" if i + 1 == len(jobs) else "running",
                       "completed_records": i + 1, "total_records": len(jobs), "job_hash": job["job_hash"],
                       "query_id": row["query_id"], "arm": arm, "utc": datetime.now(timezone.utc).isoformat()})
                if (i + 1) % 20 == 0 or i + 1 == stop:
                    print(f"{args.dataset} fixed-label records {i + 1}/{len(jobs)}", flush=True)
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
