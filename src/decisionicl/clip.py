"""Frozen CLIP image/text and corresponding-position patch representations."""
from dataclasses import dataclass
import numpy as np
from .spatial import normalize_patch_tokens


@dataclass
class ClipEncoder:
    model_path: str
    device: str = "cuda:0"
    revision: str = "local"
    batch_size: int = 32

    def __post_init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        from pathlib import Path
        from transformers import CLIPModel, CLIPProcessor
        settings = {"local_files_only": Path(self.model_path).exists()}
        if self.revision != "local":
            settings["revision"] = self.revision
        self.processor = CLIPProcessor.from_pretrained(self.model_path, **settings)
        self.model = CLIPModel.from_pretrained(self.model_path, **settings).eval().to(self.device)

    @staticmethod
    def normalized(value):
        # Recent Transformers can return a structured output.
        if not hasattr(value, "shape"):
            value = next((getattr(value, name) for name in ("image_embeds", "text_embeds", "pooler_output")
                          if getattr(value, name, None) is not None), None)
            if value is None:
                raise TypeError("unsupported CLIP feature output")
        value = value.float()
        if not bool(value.isfinite().all()):
            raise ValueError("non-finite CLIP features")
        return (value / value.norm(dim=-1, keepdim=True).clamp(min=1e-8)).cpu().numpy()

    def encode_texts(self, texts):
        self.load()
        import torch
        output = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.processor(text=list(texts[start:start+self.batch_size]), padding=True,
                                    truncation=True, max_length=77, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output.append(self.normalized(self.model.get_text_features(**inputs)))
        return np.concatenate(output)

    def encode_images(self, paths, *, spatial=False):
        self.load()
        import torch
        from PIL import Image
        output = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            try:
                for path in paths[start:start+self.batch_size]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    if spatial:
                        hidden = self.model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state
                        output.append(normalize_patch_tokens(hidden[:, 1:].float().cpu().numpy()))
                    else:
                        output.append(self.normalized(self.model.get_image_features(**inputs)))
            finally:
                for image in images:
                    image.close()
        return np.concatenate(output)

