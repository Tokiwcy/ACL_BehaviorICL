"""Independent task/generic Raw runner. Default is CPU preflight, no GPU.

Original sources and protocol locks are read-only. Cloud stages require a NEW
independent authorization, actual native evidence and fresh successful backup.
No scheduler, deployment, old-array substitution, training or width search.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import behavior_task_prompt_policy as p
from collect_behavior_priority_zero import lock

ROOT = p.ROOT
OUT = ROOT / "results" / p.SCIENCE


def sample(row, target=False):
    from multidataset_protocol import Sample
    return Sample(row["sample_id"], str(ROOT / row["path"]), row["label"] if target else "", row["split"])


def preflight(plan, dataset):
    data = plan["datasets"][dataset]
    for row in data["manifest"]["bank"] + data["manifest"]["query"]:
        if p.digest(ROOT / row["path"]) != row["image_sha256"]:
            raise ValueError("Official source image SHA changed")
    return data


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def progress(dataset, phase, **value):
    p.atomic(OUT / "sequence_progress.json", {"utc": p.now(), "dataset": dataset, "stage": phase, **value})


def source_identity(plan, binding, dataset, arm):
    data = plan["datasets"][dataset]
    return {"protocol_hash": p.PROTOCOL_HASH, "implementation_hash": binding["implementation_hash"],
            "job_hash": binding["job_hashes"][f"{dataset}/{arm}/capture"], "dataset": dataset, "arm": arm,
            "shape": [sum(p.COUNTS[dataset].values()), 36, 2560], "dtype": "float16",
            "manifest_hash": p.canonical(data["manifest"]), "origin": "fresh_paired_task_generic_not_old_sources"}


def require_compatibility(binding, dataset):
    record = p.read(OUT / dataset / "compatibility_smoke.json")
    if (record.get("status") != "actual_two_prompt_sources_and_four_demo_legal_generation_passed"
            or record.get("implementation_hash") != binding["implementation_hash"]
            or set(record.get("arms", [])) != set(p.ARMS)):
        raise ValueError("Actual two-prompt hidden/four-demo generation compatibility missing")
    diagnostic = record["diagnostic_generation"]
    p.valid_tail(diagnostic["output_token_ids"], diagnostic["prediction"], record["codec"])
    if record.get("counted_as_test_prediction") is not False or len(record.get("diagnostic_demo_ids", [])) != 4:
        raise ValueError("Diagnostic compatibility must not masquerade as official test")


def capture(plan, binding, dataset, arm, smoke=False):
    if not smoke:
        require_compatibility(binding, dataset)
    import numpy as np
    import torch
    from multimodal_model_adapter import MODEL_SPECS, load_model, model_dimensions, prepare_batch
    from prepare_behavior_task_prompt_plan import messages
    data = preflight(plan, dataset)
    rows = data["manifest"]["bank"] + data["manifest"]["query"]
    folder = OUT / dataset / "sources" / arm
    folder.mkdir(parents=True, exist_ok=True)
    identity = source_identity(plan, binding, dataset, arm)
    p.immutable(folder / "identity.json", identity)
    path, state = folder / "states.npy", folder / "progress.json"
    if path.exists() != state.exists():
        raise ValueError("Unpaired source/progress; preserve failure, no silent repair")
    if path.exists():
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        previous = p.read(state)
    else:
        array = np.lib.format.open_memmap(path, mode="w+", dtype="float16", shape=tuple(identity["shape"]))
        previous = {**identity, "completed": 0}
        p.atomic(state, previous)
    logs = p.strict_jsonl(folder / "capture.jsonl")
    count = p.capture_prefix(logs, previous, rows, identity, array)
    stop = 1 if smoke else len(rows)
    if count < stop:
        model, processor = load_model(MODEL_SPECS["qwen3vl4b"], ROOT / ".hf_cache", None)
        if model_dimensions(model) != (36, 2560):
            raise ValueError("Actual decoder layer/hidden dimensions changed")
        with (folder / "capture.jsonl").open("a", encoding="utf-8") as log, torch.inference_mode():
            for i in range(count, stop):
                row = rows[i]
                prompt = messages(sample(row), dataset, data["class_labels"], arm)
                before = time.perf_counter()
                inputs = prepare_batch(processor, [prompt])
                n = int(inputs["input_ids"].shape[1])
                if int(inputs["attention_mask"][0, -1]) != 1:
                    raise ValueError("Last-input anchor is padding")
                output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True, logits_to_keep=1)
                if output.hidden_states is None or len(output.hidden_states) != 37:
                    raise ValueError("Actual 36 pre-answer decoder states missing")
                states = torch.stack([v[0, -1].detach() for v in output.hidden_states[1:]]).float()
                if list(states.shape) != [36, 2560] or not torch.isfinite(states).all():
                    raise ValueError("Wrong/nonfinite actual pre-answer state")
                encoded = states.cpu().numpy().astype(np.float16)
                if not np.isfinite(encoded).all():
                    raise ValueError("FP16 capture became nonfinite")
                array[i] = encoded
                array.flush()
                torch.cuda.synchronize()
                entry = {"sample_id": row["sample_id"], "image_sha256": row["image_sha256"],
                         "job_hash": identity["job_hash"], "prompt_fingerprint": p.canonical(prompt),
                         "input_tokens": n, "last_input_position": n - 1, "output_tokens": 0,
                         "native_grid": inputs["image_grid_thw"].cpu().tolist(), "seconds": time.perf_counter() - before}
                write_row(log, entry)
                logs.append(entry)
                p.atomic(state, {**identity, "completed": i + 1})
                if i == 0:
                    p.immutable(folder / "smoke.json", {"status": "actual_pre_answer_36x2560_source_passed",
                                "identity": identity, "first_source_log": entry})
                progress(dataset, "fresh_source_capture", arm=arm, completed=i + 1, expected=len(rows))
                del inputs, output, states, encoded
                if (i + 1) % 100 == 0:
                    print(f"{dataset}/{arm} fresh states {i+1}/{len(rows)}", flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    if smoke:
        return
    if p.capture_prefix(logs, p.read(state), rows, identity, array) != len(rows):
        raise ValueError("Full source capture incomplete")
    p.immutable(folder / "verification.json", {"status": "all_complete_verified", "identity": identity,
                "array_sha256": p.digest(path), "capture_log_sha256": p.digest(folder / "capture.jsonl"), "rows": len(rows)})


def source(plan, binding, dataset, arm):
    import numpy as np
    folder = OUT / dataset / "sources" / arm
    expected = source_identity(plan, binding, dataset, arm)
    verification = p.read(folder / "verification.json")
    path = folder / "states.npy"
    if (verification.get("status") != "all_complete_verified" or verification.get("identity") != expected
            or verification["array_sha256"] != p.digest(path)
            or verification["capture_log_sha256"] != p.digest(folder / "capture.jsonl")):
        raise ValueError("Fresh full-source bytes/identity changed")
    data = plan["datasets"][dataset]["manifest"]
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    logs = p.strict_jsonl(folder / "capture.jsonl")
    rows = data["bank"] + data["query"]
    if (p.capture_prefix(logs, p.read(folder / "progress.json"), rows, expected, array) != len(rows)
            or verification.get("rows") != len(rows)):
        raise ValueError("Complete source has incomplete committed prefix")
    smoke = p.read(folder / "smoke.json")
    if (smoke.get("status") != "actual_pre_answer_36x2560_source_passed" or smoke.get("identity") != expected
            or smoke.get("first_source_log") != logs[0]):
        raise ValueError("Source first-row actual smoke identity changed")
    return array, verification


def retrieve(plan, binding, dataset, arm):
    import numpy as np
    import torch
    import torch.nn.functional as F
    require_compatibility(binding, dataset)
    array, verification = source(plan, binding, dataset, arm)
    data = plan["datasets"][dataset]
    bank, queries = data["manifest"]["bank"], data["manifest"]["query"]
    folder = OUT / dataset / "test" / arm
    folder.mkdir(parents=True, exist_ok=True)
    context = {"protocol_hash": p.PROTOCOL_HASH, "implementation_hash": binding["implementation_hash"],
               "job_hash": binding["job_hashes"][f"{dataset}/{arm}/generate"], "dataset": dataset, "arm": arm,
               "source_array_sha256": verification["array_sha256"], "candidate_ids": [r["sample_id"] for r in bank],
               "query_ids": [r["sample_id"] for r in queries], "retrieval_rule": plan["protocol"]["retrieval"]}
    context["context_hash"] = p.canonical(context)
    p.immutable(folder / "context.json", context)
    if (folder / "selections.json").exists():
        selections = p.read(folder / "selections.json")
        if list(selections) != context["query_ids"]:
            raise ValueError("Full official retrieval order changed")
        for selected in selections.values():
            p.validate_selection(selected, bank)
        return
    before = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    candidates = F.normalize(torch.from_numpy(np.array(array[:len(bank)], dtype=np.float32)).to("cuda"), dim=-1)
    selections = {}
    with torch.inference_mode():
        for start in range(0, len(queries), 16):
            values = np.array(array[len(bank)+start:len(bank)+min(start+16, len(queries))], dtype=np.float32)
            left = F.normalize(torch.from_numpy(values).to("cuda"), dim=-1)
            scores = torch.einsum("qld,bld->qb", left, candidates).float() / 36
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite actual Raw similarity")
            indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :4].flip(1)
            actual = torch.gather(scores, 1, indices).cpu().tolist()
            for j, (ids, saved_scores) in enumerate(zip(indices.cpu().tolist(), actual)):
                demos = [bank[i] for i in ids]
                selected = {"demo_ids": [r["sample_id"] for r in demos], "demo_labels": [r["label"] for r in demos], "scores": saved_scores}
                p.validate_selection(selected, bank)
                selections[queries[start+j]["sample_id"]] = selected
            progress(dataset, "actual_Raw_similarity", arm=arm, completed=len(selections), expected=len(queries))
            del left, scores, indices
    torch.cuda.synchronize()
    p.immutable(folder / "selections.json", selections)
    p.immutable(folder / "retrieval_time.json", {"seconds": time.perf_counter() - before, "context_hash": context["context_hash"],
                "query_count": len(queries), "scope": "offline cached FP32 normalization/scoring; frozen source extraction separately recorded"})
    del candidates
    gc.collect()
    torch.cuda.empty_cache()


def codec_for(tokenizer, labels):
    from legal_label_decoding import label_token_sequences
    sequences, maximum = label_token_sequences(tokenizer, labels)
    return {"eos_token_id": int(tokenizer.eos_token_id), "max_new_tokens": maximum,
            "label_tokens": {label: [tokenizer.encode(label, add_special_tokens=False), tokenizer.encode(" " + label, add_special_tokens=False)]
                             for label in labels}}, sequences


def actual_generation(model, processor, prompt, labels, codec, sequences):
    import torch
    from multimodal_model_adapter import prepare_batch
    from legal_label_decoding import LegalLabelTrie, exact_generated_label
    before = time.perf_counter()
    inputs = prepare_batch(processor, [prompt])
    n = int(inputs["input_ids"].shape[1])
    trie = LegalLabelTrie(sequences, codec["eos_token_id"])
    generated = model.generate(**inputs, max_new_tokens=codec["max_new_tokens"], do_sample=False, use_cache=True,
                               prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(n))
    torch.cuda.synchronize()
    ids = generated[0, n:].cpu().tolist()
    raw = processor.batch_decode(generated[:, n:], skip_special_tokens=True)[0].strip()
    prediction = exact_generated_label(raw, labels)
    p.valid_tail(ids, prediction, codec)
    result = {"prediction": prediction, "raw_answer": raw, "input_tokens": n, "output_tokens": len(ids),
              "output_token_ids": ids, "seconds": time.perf_counter() - before, "prompt_fingerprint": p.canonical(prompt)}
    del inputs, generated
    return result


def compatibility_smoke(plan, binding, dataset):
    if (OUT / dataset / "compatibility_smoke.json").exists():
        require_compatibility(binding, dataset)
        return
    import torch
    from multimodal_model_adapter import MODEL_SPECS, load_model, icl_messages
    for arm in p.ARMS:
        capture(plan, binding, dataset, arm, smoke=True)
    data = plan["datasets"][dataset]
    model, processor = load_model(MODEL_SPECS["qwen3vl4b"], ROOT / ".hf_cache", None)
    codec, sequences = codec_for(processor.tokenizer, data["class_labels"])
    demos = [sample(r, target=True) for r in data["manifest"]["bank"][:4]]
    query = sample(data["manifest"]["query"][0])
    prompt = icl_messages(demos, query, dataset, data["class_labels"], 224)
    with torch.inference_mode():
        output = actual_generation(model, processor, prompt, data["class_labels"], codec, sequences)
    # Native multi-image smoke is not a scientific official-test result and uses
    # fixed first-four bank demos, not outcome-selected retrieval or test truth.
    p.immutable(OUT / dataset / "compatibility_smoke.json", {"status": "actual_two_prompt_sources_and_four_demo_legal_generation_passed",
                "implementation_hash": binding["implementation_hash"], "arms": list(p.ARMS), "codec": codec,
                "diagnostic_demo_ids": [r["sample_id"] for r in data["manifest"]["bank"][:4]],
                "query_id": query.sample_id, "diagnostic_generation": output, "counted_as_test_prediction": False})
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def generate(plan, binding, dataset, arm, smoke=False):
    import torch
    from multimodal_model_adapter import MODEL_SPECS, load_model, icl_messages
    require_compatibility(binding, dataset)
    data = preflight(plan, dataset)
    _, source_verified = source(plan, binding, dataset, arm)
    folder = OUT / dataset / "test" / arm
    context, selections = p.read(folder / "context.json"), p.read(folder / "selections.json")
    if (context["implementation_hash"] != binding["implementation_hash"] or context["arm"] != arm
            or context["job_hash"] != binding["job_hashes"][f"{dataset}/{arm}/generate"]
            or context.get("protocol_hash") != p.PROTOCOL_HASH or context.get("dataset") != dataset
            or context["source_array_sha256"] != source_verified["array_sha256"]
            or context["candidate_ids"] != [r["sample_id"] for r in data["manifest"]["bank"]]
            or context["query_ids"] != [r["sample_id"] for r in data["manifest"]["query"]]
            or context["context_hash"] != p.canonical({k: v for k, v in context.items() if k != "context_hash"})
            or list(selections) != context["query_ids"]):
        raise ValueError("Wrong actual full retrieval context")
    for selected in selections.values():
        p.validate_selection(selected, data["manifest"]["bank"])
    model, processor = load_model(MODEL_SPECS["qwen3vl4b"], ROOT / ".hf_cache", None)
    codec, sequences = codec_for(processor.tokenizer, data["class_labels"])
    p.immutable(folder / "codec.json", codec)
    queries, bank = data["manifest"]["query"], {r["sample_id"]: r for r in data["manifest"]["bank"]}
    path = folder / "predictions.jsonl"
    rows = p.prediction_prefix(path, queries, selections, context, codec)
    if not smoke and not (folder / "smoke.json").exists():
        raise ValueError("Actual retrieved four-demo first-query smoke must precede full generation")
    stop = 1 if smoke else len(queries)
    with path.open("a", encoding="utf-8") as handle, torch.inference_mode():
        for i in range(len(rows), stop):
            query, selected = queries[i], selections[queries[i]["sample_id"]]
            demos = [sample(bank[v], target=True) for v in selected["demo_ids"]]
            prompt = icl_messages(demos, sample(query), dataset, data["class_labels"], 224)
            output = actual_generation(model, processor, prompt, data["class_labels"], codec, sequences)
            row = {**output, **selected, "query_id": query["sample_id"], "target": query["label"],
                   "correct": output["prediction"] == query["label"], "job_hash": context["job_hash"], "context_hash": context["context_hash"]}
            write_row(handle, row)
            rows.append(row)
            progress(dataset, "actual_official_test_generation", arm=arm, completed=len(rows), expected=len(queries))
            if len(rows) % 25 == 0:
                print(f"{dataset}/{arm} official ICL {len(rows)}/{len(queries)}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    p.prediction_prefix(path, queries, selections, context, codec)
    if smoke:
        p.immutable(folder / "smoke.json", {"status": "actual_retrieved_four_demo_legal_generation_passed",
                    "context_hash": context["context_hash"], "first_query_id": queries[0]["sample_id"],
                    "first_row_sha256": p.canonical(rows[0]), "first_official_query_counted_once": True})
    if len(rows) == len(queries):
        p.immutable(folder / "verification.json", {"status": "all_complete_verified", "context_hash": context["context_hash"],
                    "predictions_sha256": p.digest(path), "selections_sha256": p.digest(folder / "selections.json"),
                    "rows": len(rows), "input_tokens": sum(r["input_tokens"] for r in rows),
                    "output_tokens": sum(r["output_tokens"] for r in rows), "actual_generation_seconds": sum(r["seconds"] for r in rows)})


def verify(plan, binding):
    groups = []
    for dataset in p.DATASETS:
        require_compatibility(binding, dataset)
        data = plan["datasets"][dataset]
        for arm in p.ARMS:
            _, array_verification = source(plan, binding, dataset, arm)
            folder = OUT / dataset / "test" / arm
            context, selection = p.read(folder / "context.json"), p.read(folder / "selections.json")
            if (context["source_array_sha256"] != array_verification["array_sha256"]
                    or context["implementation_hash"] != binding["implementation_hash"]
                    or context["job_hash"] != binding["job_hashes"][f"{dataset}/{arm}/generate"]
                    or context["context_hash"] != p.canonical({k: v for k, v in context.items() if k != "context_hash"})
                    or context.get("dataset") != dataset or context.get("arm") != arm or context.get("protocol_hash") != p.PROTOCOL_HASH
                    or context["candidate_ids"] != [r["sample_id"] for r in data["manifest"]["bank"]]
                    or context["query_ids"] != [r["sample_id"] for r in data["manifest"]["query"]]
                    or list(selection) != context["query_ids"]):
                raise ValueError("Complete context/source/official cohort mismatch")
            for selected in selection.values():
                p.validate_selection(selected, data["manifest"]["bank"])
            rows = p.prediction_prefix(folder / "predictions.jsonl", data["manifest"]["query"], selection, context, p.read(folder / "codec.json"), complete=True)
            receipt, smoke = p.read(folder / "verification.json"), p.read(folder / "smoke.json")
            runtime = p.read(OUT / dataset / "runtime.json")
            timing = p.read(folder / "retrieval_time.json")
            if (receipt.get("status") != "all_complete_verified" or receipt.get("rows") != len(rows)
                    or receipt.get("context_hash") != context["context_hash"]
                    or runtime.get("implementation_hash") != binding["implementation_hash"]
                    or runtime.get("environment") != plan["expected_environment"]
                    or timing.get("context_hash") != context["context_hash"] or timing.get("query_count") != len(rows)
                    or not math.isfinite(timing.get("seconds", float("nan"))) or timing["seconds"] <= 0
                    or receipt["predictions_sha256"] != p.digest(folder / "predictions.jsonl")
                    or receipt["selections_sha256"] != p.digest(folder / "selections.json")
                    or smoke["context_hash"] != context["context_hash"] or smoke["first_row_sha256"] != p.canonical(rows[0])):
                raise ValueError("Full result byte/hash/smoke changed")
            groups.append({"dataset": dataset, "arm": arm, "rows": len(rows), "predictions_sha256": receipt["predictions_sha256"],
                           "source_array_sha256": array_verification["array_sha256"]})
    p.immutable(OUT / "verification.json", {"status": "all_complete_verified", "protocol_hash": p.PROTOCOL_HASH,
                "implementation_hash": binding["implementation_hash"], "full_groups": groups, "total_test_predictions": sum(r["rows"] for r in groups)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "smoke", "capture", "retrieve", "generate", "verify"), default="preflight")
    parser.add_argument("--dataset", choices=p.DATASETS)
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    plan = p.validate_plan(p.read(p.PLAN))
    binding = p.validate_binding(plan, p.read(p.BINDING))
    if args.phase == "preflight":
        for dataset in ([args.dataset] if args.dataset else p.DATASETS):
            preflight(plan, dataset)
        print(json.dumps({"status": "CPU_complete_images_code_protocol_job_preflight_passed_no_model_or_CUDA", "implementation_hash": binding["implementation_hash"]}))
        return
    if args.phase != "verify":
        if not args.allow_gpu or not args.dataset or (args.phase not in ("smoke",) and not args.arm):
            raise ValueError("Explicit independent cloud GPU/dataset/arm required")
        auth = p.gpu_guard(plan, binding)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / args.dataset).mkdir(parents=True, exist_ok=True)
        p.immutable(OUT / args.dataset / "runtime.json", {"protocol_hash": p.PROTOCOL_HASH,
                    "implementation_hash": binding["implementation_hash"], "pod_id": auth["pod_id"],
                    "launch_id": auth["launch_id"], "device": auth["device"], "cwd": str(ROOT),
                    "environment": plan["expected_environment"], "full_model_processor_sha_verified": True})
    elif not OUT.exists():
        raise ValueError("No completed scientific root to verify")
    OUT.mkdir(parents=True, exist_ok=True)  # exact fresh root exists before scientific lock
    with lock(OUT / "scientific_single_process.lock"):
        if args.phase == "smoke": compatibility_smoke(plan, binding, args.dataset)
        elif args.phase == "capture": capture(plan, binding, args.dataset, args.arm, args.smoke)
        elif args.phase == "retrieve": retrieve(plan, binding, args.dataset, args.arm)
        elif args.phase == "generate": generate(plan, binding, args.dataset, args.arm, args.smoke)
        elif args.phase == "verify": verify(plan, binding)


if __name__ == "__main__":
    main()
