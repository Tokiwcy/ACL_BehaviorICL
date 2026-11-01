from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import sys
from typing import Protocol, Sequence

import numpy as np


class JointEncoder(Protocol):
    model_id: str
    model_revision: str

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def logit_scale_and_bias(self) -> tuple[float, float]: ...


@dataclass
class SiglipEncoder:
    model_id: str
    model_revision: str = "local"
    device: str = "cuda"
    batch_size: int = 32
    dtype: str = "float16"

    def __post_init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            revision=None if self.model_revision == "local" else self.model_revision,
            local_files_only=Path(self.model_id).exists(),
        )
        self._model = AutoModel.from_pretrained(
            self.model_id,
            revision=None if self.model_revision == "local" else self.model_revision,
            dtype=dtype,
            local_files_only=Path(self.model_id).exists(),
        ).eval().to(self.device)

    @staticmethod
    def _normalize(features: "object") -> np.ndarray:
        import torch

        if isinstance(features, torch.Tensor):
            tensor = features
        else:
            tensor = None
            for attribute in ("image_embeds", "text_embeds", "pooler_output"):
                candidate = getattr(features, attribute, None)
                if isinstance(candidate, torch.Tensor):
                    tensor = candidate
                    break
            if tensor is None:
                raise TypeError(f"unsupported encoder feature output: {type(features).__name__}")
        tensor = tensor.float()
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        import torch
        from PIL import Image

        self._load()
        assert self._processor is not None and self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), self.batch_size):
            images = []
            for raw_path in image_paths[start : start + self.batch_size]:
                with Image.open(raw_path) as image:
                    images.append(image.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_image_features(**inputs)
            batches.append(self._normalize(features))
        if not batches:
            raise ValueError("cannot encode an empty image list")
        return np.concatenate(batches, axis=0)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        self._load()
        assert self._processor is not None and self._model is not None
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self._processor(
                text=list(texts[start : start + self.batch_size]),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self._model.get_text_features(**inputs)
            batches.append(self._normalize(features))
        if not batches:
            raise ValueError("cannot encode an empty text list")
        return np.concatenate(batches, axis=0)

    def logit_scale_and_bias(self) -> tuple[float, float]:
        import torch

        self._load()
        assert self._model is not None
        raw_scale = getattr(self._model, "logit_scale", None)
        raw_bias = getattr(self._model, "logit_bias", None)
        if raw_scale is None:
            raise RuntimeError("the loaded model does not expose SigLIP logit_scale")
        with torch.no_grad():
            scale = float(raw_scale.detach().float().exp().cpu().item())
            bias = 0.0 if raw_bias is None else float(raw_bias.detach().float().cpu().item())
        return scale, bias


