"""Bank-only target-similarity pools for the DeTriever output-proxy objective."""

from __future__ import annotations

import numpy as np


def output_proxy_candidate_indices(
    output_states: np.ndarray, positive_count: int, negative_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Select dot-product-nearest/farthest distinct bank peers, as in the paper."""
    if output_states.ndim != 2 or not np.isfinite(output_states).all():
        raise ValueError("Output proxy must be a finite bank-by-hidden-width matrix")
    count = len(output_states)
    if positive_count < 1 or negative_count < 1 or count - 1 < positive_count + negative_count:
        raise ValueError("Not enough distinct bank candidates for disjoint proxy pools")
    features = np.asarray(output_states, dtype=np.float32)
    positives = np.empty((count, positive_count), dtype=np.int64)
    negatives = np.empty((count, negative_count), dtype=np.int64)
    indices = np.arange(count)
    for start in range(0, count, 256):
        scores = features[start : start + 256] @ features.T
        for local, row in enumerate(scores):
            anchor = start + local
            row[anchor] = -np.inf
            # Stable order makes identical representation ties reproducible.
            top = np.lexsort((indices, -row))
            row[anchor] = np.inf
            bottom = np.lexsort((indices, row))
            positives[anchor] = top[:positive_count]
            negatives[anchor] = bottom[:negative_count]
    return positives, negatives
