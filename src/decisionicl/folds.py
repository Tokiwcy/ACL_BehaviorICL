from __future__ import annotations
from typing import Sequence
import numpy as np
from .schema import Sample
from .hashing import stable_hash

def stratified_folds(samples: Sequence[Sample], fold_count: int, seed: int) -> np.ndarray:
    if fold_count < 2 or fold_count > len(samples):
        raise ValueError("fold_count must be between 2 and the number of samples")
    assignments = np.full(len(samples), -1, dtype=np.int32)
    labels: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        labels.setdefault(sample.label, []).append(index)
    for label in sorted(labels):
        indices = labels[label]
        indices.sort(key=lambda index: stable_hash({"seed": seed, "sample_id": samples[index].sample_id}))
        offset = int(stable_hash({"seed": seed, "label": label})[:8], 16) % fold_count
        for rank, index in enumerate(indices):
            assignments[index] = (offset + rank) % fold_count
    if np.any(assignments < 0):
        raise RuntimeError("failed to assign every calibration sample")
    if len(set(assignments.tolist())) != fold_count:
        raise ValueError("one or more calibration folds are empty")
    return assignments
