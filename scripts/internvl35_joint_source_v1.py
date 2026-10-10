"""Shared prefill, original HF generation, and exact bank-only gold suffix."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from internvl35_joint_config_v1 import write_json


def answer_state_index(tail, eos, steps):
    positions = [i for i, token in enumerate(tail) if token == eos]
    length = max(1, positions[0] if positions else len(tail))
    return length, min(length, steps - 1)


def exact_suffix(prefix, full):
    n = prefix.shape[1]
    if full.shape[1] <= n or not torch.equal(prefix.cpu(), full[:, :n].cpu()):
        raise ValueError("Gold template is not an exact extension of the generation input")
    return full[:, n:]


@torch.inference_mode()
def extract_one(model, processor, sample, dataset, labels, *, is_bank):
    from multimodal_model_adapter import zero_shot_messages, prepare_batch
    messages = zero_shot_messages(sample, dataset, labels, None)
    inputs = prepare_batch(processor, [messages])
    prefix = inputs["input_ids"].detach().cpu()
    prompt_length = prefix.shape[1]
    generation = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                use_cache=True, return_dict_in_generate=True,
                                output_hidden_states=True)
    if generation.hidden_states is None or len(generation.hidden_states[0]) != 37:
        raise ValueError("Missing 36 decoder layers in generation prefill")
    anchors = torch.stack([h[0, -1] for h in generation.hidden_states[0][1:]]).float().cpu().numpy()
    tail = generation.sequences[0, prompt_length:].tolist()
    length, index = answer_state_index(tail, processor.tokenizer.eos_token_id, len(generation.hidden_states))
    gpt = generation.hidden_states[index][-1][0, -1].float().cpu().numpy()
    answer = processor.batch_decode(generation.sequences[:, prompt_length:], skip_special_tokens=True)[0].strip()
    cache = generation.past_key_values
    del generation
    gold = None
    suffix_ids = None
    if is_bank:
        # Keep the original template's im_end AND final newline. Only bank enters this branch.
        full = processor.apply_chat_template(
            [messages + [{"role": "assistant", "content": sample.label}]],
            tokenize=True, add_generation_prompt=False, return_dict=True,
            return_tensors="pt", processor_kwargs={"padding": True})
        suffix = exact_suffix(prefix, full["input_ids"])
        if not torch.equal(inputs["pixel_values"].cpu(), full["pixel_values"]):
            raise ValueError("Gold/prefix pixel tensors differ")
        suffix_ids = suffix[0].tolist()
        if cache is None or not hasattr(cache, "crop"):
            raise ValueError("Model returned no croppable KV cache")
        cache.crop(prompt_length)
        if cache.get_seq_length() != prompt_length:
            raise ValueError("KV crop failed")
        # Short non-square cached suffix uses math SDPA: it is small and supports
        # the explicit offset causal mask without relying on cuDNN bias support.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from internvl35_sdpa_runtime_v1 import NAME
        model.set_attn_implementation("sdpa")
        try:
            with sdpa_kernel(SDPBackend.MATH):
                output = model(input_ids=suffix.to(inputs["input_ids"].device),
                               attention_mask=full["attention_mask"].to(inputs["input_ids"].device),
                               position_ids=torch.arange(prompt_length, full["input_ids"].shape[1],
                                                         device=inputs["input_ids"].device)[None],
                               past_key_values=cache, use_cache=True, output_hidden_states=True,
                               return_dict=True, logits_to_keep=1)
        finally:
            model.set_attn_implementation(NAME)
        gold = output.hidden_states[-1][0, -1].float().cpu().numpy()
        del full, output
    values = [anchors, gpt] + ([gold] if gold is not None else [])
    if any(not np.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite joint states")
    del cache, inputs
    return {"anchor": anchors, "gpt": gpt, "gold": gold, "tail_ids": tail,
            "raw_zero_shot_answer": answer, "answer_token_count": length,
            "prefix_tokens": prompt_length,
            "prefix_ids_sha256": hashlib.sha256(prefix.numpy().tobytes()).hexdigest(),
            "gold_suffix_ids": suffix_ids}


def recover_answers(path, samples, completed):
    """Only authoritative committed rows survive a crash; truncate uncommitted tail."""
    path = Path(path)
    if not path.exists():
        if completed:
            raise ValueError("Committed answer file missing")
        path.touch()
        return
    with path.open("r+b") as handle:
        for index in range(completed):
            line = handle.readline()
            if not line.endswith(b"\n") or json.loads(line)["sample_id"] != samples[index].sample_id:
                raise ValueError("Committed answers do not match ordered manifest")
        handle.truncate(handle.tell())


def extract_dataset(model, processor, bank, query, dataset, labels, folder, binding):
    samples = bank + query
    total = len(samples)
    proxy_dir = folder / "trained/detriever"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    path = folder / "joint_progress.json"
    expected = {"binding": binding, "shape": [total, 36, 2560], "bank_n": len(bank)}
    specs = ((folder / "anchor_states.npy", (total, 36, 2560)),
             (folder / "gpt_mm_embeddings.npy", (total, 2560)),
             (proxy_dir / "gold_input_answer_eos.npy", (len(bank), 2560)))
    if path.exists():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if any(progress.get(k) != v for k, v in expected.items()):
            raise ValueError("Joint cache identity changed")
        start = progress["completed"]
        if not isinstance(start, int) or not 0 <= start <= total:
            raise ValueError("Invalid committed row cursor")
        arrays = [np.load(p, mmap_mode="r+") for p, _ in specs]
    else:
        if any(p.exists() for p, _ in specs):
            raise ValueError("Unbound arrays exist; never reuse or overwrite them")
        arrays = [np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=s) for p, s in specs]
        start = 0
        write_json(path, {**expected, "completed": 0})
    if any(a.shape != s or a.dtype != np.float16 for a, (_, s) in zip(arrays, specs)):
        raise ValueError("Invalid joint cache shape/dtype")
    answers = folder / "gpt_mm_zero_shot_outputs.jsonl"
    recover_answers(answers, samples, start)

    def publish(done):
        common = {"joint_binding": binding, "completed": done}
        write_json(folder / "anchor_states_progress.json", {**common, "shape": list(specs[0][1])})
        write_json(folder / "gpt_mm_progress.json", {**common, "shape": list(specs[1][1])})
        write_json(proxy_dir / "gold_input_answer_eos_progress.json",
                   {**common, "completed": min(done, len(bank)), "shape": list(specs[2][1]),
                    "manifest_sha256": binding["manifest_sha256"]})
    publish(start)
    with answers.open("a", encoding="utf-8") as handle:
        for i in range(start, total):
            value = extract_one(model, processor, samples[i], dataset, labels, is_bank=i < len(bank))
            arrays[0][i], arrays[1][i] = value["anchor"], value["gpt"]
            if i < len(bank):
                arrays[2][i] = value["gold"]
            for array in arrays:
                array.flush()
            row = {"sample_id": samples[i].sample_id, **{k: v for k, v in value.items() if k not in ("anchor", "gpt", "gold")}}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            write_json(path, {**expected, "completed": i + 1})
            publish(i + 1)
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
            if (i + 1) % 100 == 0 or i + 1 == total:
                print(f"{dataset}/joint_source {i + 1}/{total}; proxy {min(i + 1, len(bank))}/{len(bank)}", flush=True)
