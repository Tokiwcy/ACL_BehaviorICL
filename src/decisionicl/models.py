"""Explicit, lazily loaded target generation for the three paper model families.

This portable interface preserves raw outputs and fixed batch membership. Exact
historical replay additionally needs the released model revisions, processor
contracts and original context groups. No model is loaded merely by import.
"""
from __future__ import annotations
from dataclasses import dataclass
from .hashing import stable_hash

INVALID = "__invalid_exact_output__"


def exact_label(text, labels):
    lookup = {label.lower(): label for label in labels}
    return lookup.get(text.strip().lower(), INVALID)


def gemma_final_text(parsed):
    if type(parsed) is dict and parsed == {"role": "assistant"}:
        return ""
    if not isinstance(parsed, dict) or parsed.get("role") != "assistant":
        raise ValueError("unexpected official response schema")
    content = parsed.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(isinstance(x, dict) and x.get("type") == "text" for x in content):
        return "".join(x["text"] for x in content)
    raise ValueError("unexpected official response content")


@dataclass(frozen=True)
class ModelConfig:
    family: str
    model_path: str
    revision: str = "local"
    device: str = "cuda:0"
    max_new_tokens: int = 512

    def validate(self):
        if self.family not in {"qwen3", "qwen25", "gemma4"}:
            raise ValueError("unknown target family")
        if self.max_new_tokens < 1:
            raise ValueError("invalid generation budget")


class GenerationBackend:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.model = None
        self.processor = None

    def load(self):
        if self.model is not None:
            return
        from pathlib import Path
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        torch.backends.cuda.matmul.allow_tf32 = False
        config = self.config
        common = {"local_files_only": Path(config.model_path).exists()}
        if config.revision != "local":
            common["revision"] = config.revision
        processor_settings = {} if config.family == "gemma4" else {"min_pixels": 65536, "max_pixels": 401408}
        self.processor = AutoProcessor.from_pretrained(config.model_path, **common, **processor_settings)
        self.processor.tokenizer.padding_side = "left"
        settings = {"device_map": {"": config.device}, "dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if config.family in {"qwen25", "gemma4"}:
            settings["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        if config.family == "gemma4":
            from transformers import AutoModelForMultimodalLM
            loader = AutoModelForMultimodalLM
            if self.processor.image_processor.max_soft_tokens != 280:
                raise ValueError("Gemma processor token budget drift")
        else:
            loader = AutoModelForImageTextToText
        self.model = loader.from_pretrained(config.model_path, **common, **settings).eval()
        if {p.device.type for p in self.model.parameters()} != {"cuda"}:
            raise ValueError("the paper target protocol forbids CPU offload")

    def generate(self, conversations, *, max_new_tokens=None):
        self.load()
        import torch
        kwargs = {"enable_thinking": False} if self.config.family == "gemma4" else {}
        inputs = self.processor.apply_chat_template(conversations, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True}, **kwargs).to(self.config.device, dtype=torch.bfloat16)
        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, use_cache=True,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens)
        generated = output[:, width:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=False)
        if self.config.family == "gemma4":
            eos = self.model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos)
            texts = []
            for row, prefix in zip(generated, inputs["input_ids"]):
                values = row.tolist()
                end = next((i+1 for i, token in enumerate(values) if token in eos), len(values))
                texts.append(gemma_final_text(self.processor.parse_response(values[:end], prefix=prefix)))
        else:
            texts = self.processor.batch_decode(generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
        group_hash = stable_hash({"config": self.config.__dict__, "conversations": conversations})
        return [{"text": text.strip(), "raw_output": original,
                 "batch_group_hash": group_hash, "batch_row": i, "batch_size": len(conversations)}
                for i, (text, original) in enumerate(zip(texts, raw))]

    def classify(self, conversations, labels):
        return [{**record, "prediction": exact_label(record["text"], labels)}
                for record in self.generate(conversations)]


class FactorDiscoveryBackend:
    """Adapter for discovery.make_discovery_requests / parse_discovery_response."""
    def __init__(self, generation_backend):
        self.backend = generation_backend
        self.model_id = generation_backend.config.model_path

    def generate(self, image_paths, prompt):
        conversation = [{"role": "user", "content":
            [{"type": "image", "image": str(path)} for path in image_paths]
            + [{"type": "text", "text": prompt}]}]
        return self.backend.generate([conversation], max_new_tokens=3072)[0]["text"]

