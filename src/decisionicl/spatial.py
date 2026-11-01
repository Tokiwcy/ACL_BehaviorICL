"""Prompt-SelF classification adaptation: corresponding-position CLIP patches."""
from __future__ import annotations
import numpy as np
PATCH_COUNT = 256
CHANNELS = 1024

def normalize_patch_tokens(tokens: np.ndarray) -> np.ndarray:
    """Normalize each spatial position across channels (Algorithm 1 dim=1)."""
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (PATCH_COUNT, CHANNELS):
        raise ValueError(f"Expected (images,256,1024) patch tokens, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite patch features")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Zero patch feature")
    return values / norm


def spatial_scores(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Mean corresponding-position cosine, query by full training pool."""
    q = normalize_patch_tokens(query)
    t = normalize_patch_tokens(train)
    return np.einsum("qpc,tpc->qt", q, t, optimize=True) / PATCH_COUNT


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite query-by-train scores")
    if not 1 <= k <= values.shape[1]:
        raise ValueError("Invalid k")
    return np.argsort(-values, axis=1, kind="stable")[:, :k]
