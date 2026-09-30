"""Shared Qwen3-VL/Gemma-3 message, hidden-state, and generation adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from legal_label_decoding import LegalLabelTrie, exact_generated_label, label_token_sequences
from multidataset_protocol import Sample


TASK_NOUNS = {
    "dtd": "texture category",
    "aircraft": "aircraft variant",
    "cub": "bird species",
    "dogs": "dog breed",
    "pets": "cat or dog breed",
}


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    image_size: int | None


MODEL_SPECS = {
    "qwen3vl4b": ModelSpec("qwen3vl4b", "Qwen/Qwen3-VL-4B-Instruct", 224),
    "gemma3_4b": ModelSpec("gemma3_4b", "google/gemma-3-4b-it", None),
}


def image_part(path: str, image_size: int | None) -> dict:
    result = {"type": "image", "image": path}
    if image_size is not None:
        result.update({"resized_height": image_size, "resized_width": image_size})
    return result


def task_instruction(dataset: str, labels: list[str]) -> str:
    noun = TASK_NOUNS[dataset]
    return (
        f"Classify the image into exactly one {noun}. "
        f"The only valid labels are: {', '.join(labels)}. "
        "Return exactly one valid label and no explanation. Label:"
    )


def zero_shot_messages(
    sample: Sample, dataset: str, labels: list[str], image_size: int | None
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [image_part(sample.path, image_size), {"type": "text", "text": task_instruction(dataset, labels)}],
        }
    ]


def icl_messages(
    demos: list[Sample], sample: Sample, dataset: str, labels: list[str], image_size: int | None
) -> list[dict]:
    noun = TASK_NOUNS[dataset]
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Learn the image-to-{noun} task from the demonstrations. "
                f"The only valid labels are: {', '.join(labels)}.\n"
            ),
        }
    ]
    for number, demo in enumerate(demos, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Demonstration {number}:"},
                image_part(demo.path, image_size),
                {"type": "text", "text": f"Label: {demo.label}\n"},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "Query image:"},
            image_part(sample.path, image_size),
            {"type": "text", "text": "Return exactly one valid label and no explanation. Label:"},
        ]
    )
    return [{"role": "user", "content": content}]


def model_dimensions(model) -> tuple[int, int]:
    config = getattr(model.config, "text_config", model.config)
    return int(config.num_hidden_layers), int(config.hidden_size)


def move_inputs(inputs: dict, device: str = "cuda") -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def configure_vision_pixels(processor, pixels: int | None) -> None:
    """Apply an explicit Qwen image-area budget to the actual image processor.

    Per-message resized_height/width hints are ignored by the Transformers
    apply_chat_template path used here, so those hints are not a pixel limit.
    """
    if pixels is None:
        return
    if pixels < 1024 or pixels % 1024:
        raise ValueError("vision_pixels must be a positive multiple of 1024")
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not hasattr(image_processor, "size"):
        raise TypeError("This processor does not expose a configurable image size")
    image_processor.size = {"shortest_edge": pixels, "longest_edge": pixels}


def load_model(spec: ModelSpec, cache_dir: Path, vision_pixels: int | None = None):
    if vision_pixels is not None and spec.slug != "qwen3vl4b":
        raise ValueError("Explicit vision_pixels is currently supported only for Qwen3-VL")
    processor = AutoProcessor.from_pretrained(spec.model_id, cache_dir=cache_dir)
    configure_vision_pixels(processor, vision_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id, cache_dir=cache_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().to("cuda")
    return model, processor


def prepare_batch(processor, messages_batch: list[list[dict]]) -> dict:
    previous = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
    finally:
        processor.tokenizer.padding_side = previous
    return move_inputs(inputs)


@torch.inference_mode()
def anchor_hidden_states(model, inputs: dict) -> torch.Tensor:
    layer_count, _ = model_dimensions(model)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    states = outputs.hidden_states
    if states is None or len(states) != layer_count + 1:
        raise RuntimeError(
            f"Unexpected hidden-state structure: expected {layer_count + 1}, "
            f"got {None if states is None else len(states)}"
        )
    return torch.stack([state[:, -1].detach() for state in states[1:]], dim=1)


@torch.inference_mode()
def generate_legal_labels(
    model,
    processor,
    messages_batch: list[list[dict]],
    labels: list[str],
) -> list[tuple[str, str]]:
    inputs = prepare_batch(processor, messages_batch)
    prompt_length = int(inputs["input_ids"].shape[1])
    sequences, max_new_tokens = label_token_sequences(processor.tokenizer, labels)
    trie = LegalLabelTrie(sequences, processor.tokenizer.eos_token_id)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        prefix_allowed_tokens_fn=trie.prefix_allowed_tokens_fn(prompt_length),
    )
    tails = generated[:, prompt_length:]
    texts = [
        value.strip()
        for value in processor.batch_decode(tails, skip_special_tokens=True)
    ]
    result = [(exact_generated_label(text, labels), text) for text in texts]
    del inputs, generated, tails
    return result
