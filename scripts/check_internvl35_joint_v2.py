"""Predeclared GPU extraction parity pilot; never score test accuracy."""
from __future__ import annotations
import argparse
import hashlib
import json
import numpy as np
import torch
from internvl35_joint_config_v2 import *


def pilot(samples, dataset, n):
    return sorted(samples, key=lambda s: hashlib.sha256(f"{dataset}/{s.sample_id}".encode()).hexdigest())[:n]


def cosine_min(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(np.min(np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))))


def ordered_top4(states, bank_n, raw=False):
    x = np.asarray(states, dtype=np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    if raw:
        scores = np.einsum('qld,bld->qb', x[bank_n:], x[:bank_n]) / x.shape[1]
    else:
        scores = x[bank_n:] @ x[:bank_n].T
    return np.argsort(scores, axis=1)[:, -4:].tolist()


@torch.inference_mode()
def reference(model, processor, sample, dataset, labels, is_bank):
    from multimodal_model_adapter import prepare_batch, zero_shot_messages, anchor_hidden_states
    from run_detriever_output_proxy import gold_answer_eos_state
    from internvl35_joint_source_v2 import answer_state_index
    inputs = prepare_batch(processor, [zero_shot_messages(sample, dataset, labels, None)])
    anchor = anchor_hidden_states(model, inputs)[0].float().cpu().numpy()
    n = inputs["input_ids"].shape[1]
    gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True,
                         return_dict_in_generate=True, output_hidden_states=True)
    tail = gen.sequences[0, n:].tolist()
    _, i = answer_state_index(tail, processor.tokenizer.eos_token_id, len(gen.hidden_states))
    gpt = gen.hidden_states[i][-1][0, -1].float().cpu().numpy()
    del gen, inputs
    gold = gold_answer_eos_state(model, processor, sample, dataset, labels, None).numpy() if is_bank else None
    return {"anchor": anchor, "gpt": gpt, "gold": gold, "tail_ids": tail}


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_protocol()
    from run_internvl35_qwen_pipeline_v4 import gpu_free
    gpu_free()
    path = ANALYSIS / "compatibility.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("state") == "complete":
            require_gate()
            print("Exact binding already passed; no repeated GPU pilot", flush=True)
            return
        raise RuntimeError("Previous GPU pilot did not complete; diagnose before explicit retry")
    report = {"state": "running", "started_utc": now(), "protocol_sha256": digest(PROTOCOL_PATH),
              "cases": {}, "threshold_cosine": .9999, "test_accuracy_used": False}
    write_json(path, report)
    try:
        from run_internvl35_joint_methods_v2 import load_context
        from internvl35_joint_source_v2 import extract_one
        model, processor = load_model(ANALYSIS / "compatibility_inputs.jsonl")
        torch.cuda.reset_peak_memory_stats()
        for dataset in DATASETS:
            folder, bank, query, _ = load_context(dataset)
            labels = sorted({s.label for s in bank})
            chosen = pilot(bank, dataset, 8) + pilot(query, dataset, 4)
            olds, news, cases = [], [], []
            for i, sample in enumerate(chosen):
                old = reference(model, processor, sample, dataset, labels, i < 8)
                calls = []
                # Count actual vision encodings, rather than inferring them from processor logs.
                hook = model.model.vision_tower.register_forward_hook(lambda *args: calls.append(1))
                try:
                    new = extract_one(model, processor, sample, dataset, labels, is_bank=i < 8)
                finally:
                    hook.remove()
                entry = {"sample_id": sample.sample_id, "is_bank": i < 8,
                         "tail_ids_exact": old["tail_ids"] == new["tail_ids"],
                         "vision_forwards": len(calls), "expected_vision_forwards": 2 if i < 8 else 1, "prefix_tokens": new["prefix_tokens"],
                         "gold_suffix_ids": new["gold_suffix_ids"],
                         "cosine_min": {k: cosine_min(old[k], new[k]) for k in ("anchor", "gpt", "gold") if old[k] is not None},
                         "max_abs": {k: float(np.max(np.abs(old[k] - new[k]))) for k in ("anchor", "gpt", "gold") if old[k] is not None}}
                cases.append(entry)
                report["cases"][dataset] = {"rows": cases}
                write_json(path, report)
                if len(calls) != (2 if i < 8 else 1) or not entry["tail_ids_exact"] or min(entry["cosine_min"].values()) < .9999:
                    raise ValueError(f"Extraction parity failed: {dataset}/{sample.sample_id}: {entry}")
                olds.append(old)
                news.append(new)
                print(f"parity {dataset} {i+1}/12: {entry['cosine_min']}", flush=True)
                torch.cuda.empty_cache()
            ranks = {}
            for key in ("anchor", "gpt"):
                ranks[key] = ordered_top4([x[key] for x in olds], 8, key == "anchor") == ordered_top4([x[key] for x in news], 8, key == "anchor")
            def proxy_ranks(rows):
                x = np.asarray([r["gold"] for r in rows[:8]], dtype=np.float32)
                x /= np.linalg.norm(x, axis=-1, keepdims=True)
                score = x @ x.T
                np.fill_diagonal(score, -np.inf)
                return np.argsort(score, axis=1)[:, -4:].tolist()
            ranks["proxy_bank"] = proxy_ranks(olds) == proxy_ranks(news)
            report["cases"][dataset]["ordered_top4_exact"] = ranks
            if not all(ranks.values()):
                raise ValueError(f"Pilot top4 parity failed: {dataset}: {ranks}")
            write_json(path, report)
        report.update(state="complete", finished_utc=now(),
                      peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30,
                      limits="36 fixed pilot rows; finite BF16 tolerance and pilot ranks do not prove all-bank rank equality")
        write_json(path, report)
    except Exception as error:
        report.update(state="failed", error=repr(error), finished_utc=now())
        write_json(path, report)
        raise

if __name__ == "__main__":
    main()
