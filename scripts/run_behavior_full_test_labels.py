"""Independent full official DTD+CUB label runner; original sample is read-only.

Uses only frozen original pure prompt/record helpers, not their capture main or
sampled authorization. Default is complete CPU input/image preflight.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

import behavior_task_prompt_policy as common
import run_behavior_label_intervention as original
from collect_behavior_priority_zero import lock

ROOT = common.ROOT
DATASETS = ("dtd", "cub")
COUNTS = {"dtd": 1880, "cub": 5794}
MISSING = {"dtd": 71, "cub": 99}
SCIENCE = "behavior_corrective_labels_full_dtd_cub_v1"
OPERATION = "behavior_corrective_labels_full_dtd_cub_cloud_v1"
PLAN = ROOT / "analysis/paired_label_full_test_dtd_cub_plan_v1/query_arm_lock.json"
INPUT = PLAN.with_name("attached_inputs_lock.json")
BINDING = ROOT / "analysis/paired_label_full_test_implementation_v1/implementation_lock.json"
PLAN_HASH = "6ce80bf47d662198c9d2f593bce04cf83ddc04791dffa07401eb8f3a82f48096"
INPUT_HASH = "7c4a79175241bedb7ea982fab65c03eaceaf19fcf8ace318d197699b543505f7"
OLD_RUNNER_SHA = "50d2d2378e43ffff7d3614d5565b917bdd50bd7af51645da9534b502ab7c6580"


def validate(plan, inputs):
    if (plan.get("plan_hash") != PLAN_HASH or common.canonical({k: v for k, v in plan.items() if k != "plan_hash"}) != PLAN_HASH
            or inputs.get("base_plan_hash") != PLAN_HASH or inputs.get("attachment_hash") != INPUT_HASH
            or common.canonical({k: v for k, v in inputs.items() if k != "attachment_hash"}) != INPUT_HASH
            or list(plan.get("datasets", [])) != list(DATASETS) or plan.get("reuse_sample_predictions") is not False
            or inputs.get("paired_arms") != list(original.ARMS) or plan.get("paired_arms") != list(original.ARMS)
            or inputs.get("revision") != "ebb281ec70b05090aa6165b016eac8ec08e71b17"
            or common.digest(Path(original.__file__)) != OLD_RUNNER_SHA):
        raise ValueError("Independent full-test lock or original sampled runner changed")
    if len(inputs["all_queries"]) != 7674 or len(plan["all_queries"]) != 7674:
        raise ValueError("Complete official cohort required")
    for dataset in DATASETS:
        rows = [r for r in inputs["all_queries"] if r["dataset"] == dataset]
        manifest = plan["manifest_with_image_SHA"][dataset]
        bank = {r["sample_id"]: r for r in manifest["bank"]}
        queries = manifest["query"]
        if (len(rows) != COUNTS[dataset] or len({r["query_id"] for r in rows}) != COUNTS[dataset]
                or [r["query_id"] for r in rows] != [r["sample_id"] for r in queries]
                or sum(r["own_arm_status"] != "ready" for r in rows) != MISSING[dataset]):
            raise ValueError("Full query order/count or explicit missing denominator changed")
        for row, query in zip(rows, queries):
            ids = row["demo_ids_in_actual_prompt_order"]
            if (len(ids) != 4 or len(set(ids)) != 4 or not set(ids) <= set(bank)
                    or row["target_for_evaluation_only"] != query["label"]
                    or row["gold_labels"] != [bank[i]["label"] for i in ids]
                    or row["query_image"]["sample_id"] != query["sample_id"]
                    or row["query_image"]["sha256"] != query["image_sha256"]
                    or row["legal_classes"] != sorted({r["label"] for r in manifest["bank"]})):
                raise ValueError("Fixed query/demo/gold/image context changed")
            own = []
            legal = {v.casefold(): v for v in row["legal_classes"]}
            for sample_id, image, gold, evidence, wrong in zip(ids, row["demo_images_in_actual_prompt_order"],
                    row["gold_labels"], row["own_zero_evidence"], row["hashed_wrong_labels"]):
                actual = legal.get(evidence["raw_zero_shot_answer"].strip().casefold())
                if (image["sample_id"] != sample_id or image["sha256"] != bank[sample_id]["image_sha256"]
                        or evidence["demo_id"] != sample_id or evidence["gold_label"] != gold
                        or evidence["predicted_label"] != actual or evidence["valid"] != (actual is not None)
                        or wrong not in row["legal_classes"] or wrong == gold):
                    raise ValueError("Frozen demo image/free-zero/label intervention changed")
                own.append(actual)
            if (len(own) != 4 or row["demo_own_zero_labels"] != own
                    or row["own_arm_status"] != ("ready" if all(v is not None for v in own) else "missing_illegal_demo_zero")):
                raise ValueError("Explicit illegal own-label missingness changed")
    for a, b in zip(inputs["all_queries"], plan["all_queries"]):
        for key, value in b.items():
            if key not in ("demo_own_zero_labels", "own_arm_status") and a.get(key) != value:
                raise ValueError("Locked base context differs from attached full input")


def job(plan, inputs, binding, dataset):
    if dataset not in DATASETS:
        raise ValueError("Wrong prespecified dataset")
    body = {"experiment": SCIENCE, "dataset": dataset, "base_plan_hash": PLAN_HASH, "attachment_hash": INPUT_HASH,
            "official_query_ids": [r["query_id"] for r in inputs["all_queries"] if r["dataset"] == dataset],
            "arm_order": list(original.ARMS), "code_sha256": binding["code_sha256"],
            "model_inventory": plan["model_weight_and_processor_inventory"], "expected_environment": plan["expected_environment"],
            "generation": plan["generation"], "own_missing_policy": plan["own_invalid_policy"],
            "reuse_sample_predictions": False}
    body["job_hash"] = common.canonical(body)
    return body


def binding_check(plan, inputs, binding):
    if (common.canonical({k: v for k, v in binding.items() if k != "implementation_hash"}) != binding["implementation_hash"]
            or binding["plan_hash"] != PLAN_HASH or binding["attachment_hash"] != INPUT_HASH
            or binding["plan_file_sha256"] != common.digest(PLAN) or binding["input_file_sha256"] != common.digest(INPUT)
            or binding["scientific_root"] != SCIENCE or binding["operational_root"] != OPERATION):
        raise ValueError("Independent full-label binding changed")
    for rel, sha in binding["code_sha256"].items():
        if common.digest(ROOT / rel) != sha:
            raise ValueError("Independent full-label code changed: " + rel)
    if binding["job_hashes"] != {d: job(plan, inputs, binding, d)["job_hash"] for d in DATASETS}:
        raise ValueError("Independent full-test scientific job changed")


def smoke_count(rows):
    generated = set()
    for n, (row, arm) in enumerate(original.schedule(rows), 1):
        if original.expected_record(row, arm, "smoke")["status"] == "generated":
            generated.add(arm)
        if generated == set(original.ARMS):
            return n
    raise ValueError("No actual four-arm smoke cohort exists")


def backup_guard(auth):
    folder = ROOT / "results" / OPERATION
    common.native_identity(auth["pod_id"])
    common.validate_backups(auth, common.read(folder / "backup_ready.json"), common.read(folder / "backup_heartbeat.json"))


def cloud_guard(plan, binding):
    if sys.platform != "linux" or str(ROOT) != "/workspace/BehaviorICL" or Path.cwd() != ROOT:
        raise ValueError("Independent exact cloud full-label execution only; no local CUDA")
    folder = ROOT / "results" / OPERATION
    auth = common.read(folder / "execution_authorization.json")
    if (auth.get("experiment") != SCIENCE or auth.get("protocol_hash") != PLAN_HASH
            or auth.get("implementation_hash") != binding["implementation_hash"]
            or auth.get("actual_api_pod_id") != auth.get("pod_id") or not auth.get("pod_id")
            or auth.get("code_sha256") != binding["code_sha256"]):
        raise ValueError("Wrong fresh independently audited full-label authorization")
    proof = common.read(folder / "native_identity_receipt.json")
    if (common.digest(folder / "native_identity_receipt.json") != auth.get("native_identity_receipt_sha256")
            or proof.get("actual_api_pod_id") != auth["pod_id"]
            or not any(proof.get(k) == auth["pod_id"] for k in ("proc1_pod_id", "export_file_pod_id"))
            or any(proof.get(k) not in (None, auth["pod_id"]) for k in ("proc1_pod_id", "export_file_pod_id", "ssh_process_pod_id"))
            or (proof.get("export_file_pod_id") is not None and not proof.get("export_root_owned_nonwritable_regular"))):
        raise ValueError("Actual native full-label proof absent/conflicting/untrusted")
    backup_guard(auth)
    import subprocess
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if any(v.strip().isdigit() and int(v.strip()) != os.getpid() for v in active.splitlines()):
        raise ValueError("Another GPU compute process exists")
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != auth.get("device")
            or not any(v in auth["device"] for v in ("4090", "5090", "RTX PRO 4500"))):
        raise ValueError("Wrong full-label authorized single GPU")
    if {k: importlib.metadata.version(k) for k in plan["expected_environment"]} != plan["expected_environment"]:
        raise ValueError("Pinned full-label environment changed")
    snapshot = ROOT / ".hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / plan["revision"]
    for name, sha in plan["model_weight_and_processor_inventory"]["files_SHA256"].items():
        if common.digest(snapshot / name) != sha:
            raise ValueError("Fixed full model/processor SHA changed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return auth


def verified_prefix(path, rows, identity, codec=None, complete=False):
    records = original.validate_prefix(path, original.schedule(rows), identity["job_hash"])
    if complete and len(records) != 4 * len(rows):
        raise ValueError("Incomplete full official intervention; no partial accuracy")
    if codec:
        for record in records:
            if record["status"] == "generated":
                common.valid_tail(record["generated_token_ids"], record["prediction"], codec)
    return records


def capture(plan, inputs, binding, dataset, smoke=False):
    auth = cloud_guard(plan, binding)
    rows = [r for r in inputs["all_queries"] if r["dataset"] == dataset]
    original.verify_images(rows)
    folder = ROOT / "results" / SCIENCE / dataset
    folder.mkdir(parents=True, exist_ok=True)
    identity = job(plan, inputs, binding, dataset)
    common.immutable(folder / "identity.json", identity)
    if not smoke and not (folder / "actual_smoke.json").exists():
        raise ValueError("Real four-arm smoke required before full official generation")
    from run_behavior_task_prompt import codec_for
    from multimodal_model_adapter import load_model, MODEL_SPECS, prepare_batch
    from legal_label_decoding import LegalLabelTrie, exact_generated_label
    import torch
    before = time.perf_counter()
    model, processor = load_model(MODEL_SPECS["qwen3vl4b"], ROOT / ".hf_cache", None)
    torch.cuda.synchronize()
    common.immutable(folder / ("runtime_" + ("smoke" if smoke else "full") + ".json"),
                     {"pod_id": auth["pod_id"], "launch_id": auth["launch_id"], "device": auth["device"],
                      "implementation_hash": binding["implementation_hash"], "job_hash": identity["job_hash"],
                      "environment": plan["expected_environment"], "model_load_seconds": time.perf_counter() - before})
    codec, sequences = codec_for(processor.tokenizer, rows[0]["legal_classes"])
    common.immutable(folder / "codec.json", codec)
    trie = LegalLabelTrie(sequences, codec["eos_token_id"])
    path = folder / "predictions.jsonl"
    records = verified_prefix(path, rows, identity, codec)
    jobs = original.schedule(rows)
    end = smoke_count(rows) if smoke else len(jobs)
    with path.open("a", encoding="utf-8") as handle:
        for i in range(len(records), end):
            backup_guard(auth)
            row, arm = jobs[i]
            record = original.expected_record(row, arm, identity["job_hash"])
            if record["status"] != "generated":
                record.update(raw_answer=None, prediction=None, correct=None, input_tokens=0, output_tokens=0,
                              elapsed_seconds=0, demo_label_token_counts=None)
            else:
                messages = original.render_messages(row, arm)
                torch.cuda.synchronize()
                before = time.perf_counter()
                batch = prepare_batch(processor, [messages])
                n = int(batch["input_ids"].shape[1])
                with torch.inference_mode():
                    output = model.generate(**batch, max_new_tokens=codec["max_new_tokens"], do_sample=False, use_cache=True,
                                            prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(n))
                torch.cuda.synchronize()
                tail = output[0, n:].cpu().tolist()
                raw = processor.batch_decode(output[:, n:], skip_special_tokens=True)[0].strip()
                pred = exact_generated_label(raw, row["legal_classes"])
                common.valid_tail(tail, pred, codec)
                record.update(raw_answer=raw, prediction=pred, correct=pred == row["target_for_evaluation_only"],
                              input_tokens=n, output_tokens=len(tail), generated_token_ids=tail,
                              elapsed_seconds=time.perf_counter() - before, prompt_messages_sha256=common.canonical(messages),
                              demo_label_token_counts=[len(processor.tokenizer.encode(" " + v, add_special_tokens=False)) for v in record["demo_labels"]])
                del batch, output
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            records.append(record)
            common.atomic(folder / "progress.json", {"job_hash": identity["job_hash"], "completed_records": len(records),
                          "total_records": len(jobs), "utc": common.now()})
            if len(records) % 25 == 0:
                print(f"{dataset} full-test label records {len(records)}/{len(jobs)}", flush=True)
    verified_prefix(path, rows, identity, codec)
    if smoke:
        if not all(any(r["arm"] == arm and r["status"] == "generated" for r in records) for arm in original.ARMS):
            raise ValueError("Actual four-arm generation smoke incomplete")
        common.immutable(folder / "actual_smoke.json", {"status": "actual_generated_all_four_arms", "job_hash": identity["job_hash"],
                         "prefix_records": end, "prefix_hash": common.canonical(records[:end]), "counted_once_as_official_prefix": True})


def verify(plan, inputs, binding, dataset):
    rows = [r for r in inputs["all_queries"] if r["dataset"] == dataset]
    folder = ROOT / "results" / SCIENCE / dataset
    identity = job(plan, inputs, binding, dataset)
    if common.read(folder / "identity.json") != identity:
        raise ValueError("New full-test identity changed")
    records = verified_prefix(folder / "predictions.jsonl", rows, identity, common.read(folder / "codec.json"), complete=True)
    smoke = common.read(folder / "actual_smoke.json")
    if (smoke["job_hash"] != identity["job_hash"] or smoke["prefix_hash"] != common.canonical(records[:smoke_count(rows)])
            or sum(r["status"] != "generated" for r in records) != MISSING[dataset]):
        raise ValueError("Actual smoke/missingness changed")
    runtime = {v.name: common.read(v) for v in folder.glob("runtime_*.json")}
    if not runtime:
        raise ValueError("Actual runtime receipt absent")
    for value in runtime.values():
        if (value.get("implementation_hash") != binding["implementation_hash"] or value.get("job_hash") != identity["job_hash"]
                or value.get("environment") != plan["expected_environment"] or not value.get("pod_id")
                or not math.isfinite(value.get("model_load_seconds", float("nan"))) or value["model_load_seconds"] <= 0):
            raise ValueError("Actual pinned runtime evidence changed")
    common.immutable(folder / "verification.json", {"status": "all_complete_verified", "job_hash": identity["job_hash"],
                     "queries": len(rows), "records": len(records), "generated": len(records) - MISSING[dataset],
                     "explicit_missing": MISSING[dataset], "outputs_sha256": common.digest(folder / "predictions.jsonl"),
                     "input_tokens": sum(r["input_tokens"] for r in records), "output_tokens": sum(r["output_tokens"] for r in records),
                     "elapsed_seconds": sum(r["elapsed_seconds"] for r in records)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "capture", "verify"), default="preflight")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    plan, inputs, binding = common.read(PLAN), common.read(INPUT), common.read(BINDING)
    validate(plan, inputs)
    binding_check(plan, inputs, binding)
    rows = [r for r in inputs["all_queries"] if r["dataset"] == args.dataset]
    if args.phase == "preflight":
        count = original.verify_images(rows)
        print(json.dumps({"status": "CPU_full_official_label_preflight_not_GPU", "dataset": args.dataset,
                          "queries": len(rows), "planned_generations": 4 * len(rows) - MISSING[args.dataset],
                          "explicit_missing": MISSING[args.dataset], "images_verified": count,
                          "job_hash": job(plan, inputs, binding, args.dataset)["job_hash"]}))
        return
    if args.phase == "capture" and not args.allow_gpu:
        raise ValueError("Explicit independent GPU permission required")
    folder = ROOT / "results" / SCIENCE
    if args.phase == "capture":
        cloud_guard(plan, binding)
        folder.mkdir(parents=True, exist_ok=True)
    with lock(folder / "scientific_single_process.lock"):
        if args.phase == "capture":
            capture(plan, inputs, binding, args.dataset, args.smoke)
        else:
            verify(plan, inputs, binding, args.dataset)


if __name__ == "__main__":
    main()
