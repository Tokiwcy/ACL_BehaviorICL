"""Paper-specific retrieval parents and controls over the full training pool.

Ranking rules are explicit: main RICES/MUIER use reverse NumPy argsort;
CIRCLES takes the last tied maximum; Generic/Prompt-SelF use stable order.
"""
from __future__ import annotations
import numpy as np
from .selection import class_features


def topk(scores, shots, *, stable=False):
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite query-by-train scores")
    if not 1 <= shots <= values.shape[1]:
        raise ValueError("invalid demonstration budget")
    if stable:
        return np.argsort(-values, axis=1, kind="stable")[:, :shots]
    return np.argsort(values, axis=1)[:, -shots:][:, ::-1]


def random_demonstrations(query_count, train_count, shots, seed=42):
    if not 1 <= shots <= train_count:
        raise ValueError("invalid demonstration budget")
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(train_count, shots, replace=False)
                       for _ in range(query_count)], dtype=np.int64)


def rices_scores(query_embeddings, train_embeddings):
    """CLIP L2-normalized image embeddings, no score rescaling."""
    return np.asarray(query_embeddings, np.float32) @ np.asarray(train_embeddings, np.float32).T


def muier_scores(query_embeddings, train_embeddings, train_y, label_embeddings):
    """Accepted classification adaptation: SigLIP image cosine + train-label cosine."""
    image = rices_scores(query_embeddings, train_embeddings)
    labels = np.asarray(query_embeddings, np.float32) @ np.asarray(label_embeddings, np.float32).T
    return image + labels[:, np.asarray(train_y, np.int64)]


def last_argmax(scores):
    values = np.asarray(scores)
    return values.shape[1] - 1 - np.argmax(values[:, ::-1], axis=1)


def circles_features(image_scores, caption_scores, train_y, factor_similarities):
    """Mean of channel-wise class maxima, then standardize each feature across classes."""
    image = np.asarray(image_scores, np.float32)
    caption = np.asarray(caption_scores, np.float32)
    y = np.asarray(train_y)
    factors = np.asarray(factor_similarities, np.float32)
    if image.shape != caption.shape or image.shape[1] != len(y):
        raise ValueError("CIRCLES channels must cover the same full train pool")
    first = np.empty(factors.shape[:2], np.int64)
    second = np.empty_like(first)
    parent = np.empty(factors.shape[:2], np.float32)
    rows = np.arange(len(image))
    for cls in range(factors.shape[1]):
        candidates = np.flatnonzero(y == cls)
        if not len(candidates):
            raise ValueError("class has no train candidates")
        first[:, cls] = candidates[last_argmax(image[:, candidates])]
        second[:, cls] = candidates[last_argmax(caption[:, candidates])]
        parent[:, cls] = .5 * (image[rows, first[:, cls]] + caption[rows, second[:, cls]])
    raw = np.concatenate((parent[:, :, None], factors), axis=2)
    features = (raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True).clip(min=1e-6)
    return features.astype(np.float32), first, second


def circles_demonstrations(image_scores, caption_scores, *, features=None, weights=None,
                           train_y=None, factor_similarities=None):
    """Exactly two channels; duplicate images across channels are retained."""
    if weights is None:
        return np.stack((last_argmax(image_scores), last_argmax(caption_scores)), axis=1)
    built, first, second = circles_features(image_scores, caption_scores, train_y, factor_similarities)
    if features is None:
        features = built
    selected = np.einsum("qcf,f->qc", features, weights).argmax(1)
    rows = np.arange(len(selected))
    return np.stack((first[rows, selected], second[rows, selected]), axis=1)


def base_class_gate(scores, train_y, shots):
    """No-factor ablation: choose top-one base class and retrieve within it."""
    values = np.asarray(scores)
    y = np.asarray(train_y)
    chosen_class = y[topk(values, 1)[:, 0]]
    result = []
    for row, cls in enumerate(chosen_class):
        candidates = np.flatnonzero(y == cls)
        result.append(candidates[topk(values[row:row+1, candidates], shots)[0]])
    return np.asarray(result)


def within_class(scores, train_y, selected_classes, shots, *, stable=False):
    """Rank the selected class's complete pool, with the parent's frozen tie rule."""
    y = np.asarray(train_y)
    result = []
    for i, cls in enumerate(selected_classes):
        candidates = np.flatnonzero(y == cls)
        result.append(candidates[topk(np.asarray(scores)[i:i+1, candidates], shots, stable=stable)[0]])
    return np.asarray(result)


def centroid_factor(query_embeddings, train_embeddings, train_y, class_count):
    """Normalized CLIP train-class centroids as one calibrated alternative feature."""
    y = np.asarray(train_y)
    centers = np.stack([np.asarray(train_embeddings)[y == c].mean(0) for c in range(class_count)])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    return rices_scores(query_embeddings, centers)[:, :, None]


def rotated_factor_features(scores, train_y, factors):
    """Frozen source control: prototype class correspondence shifted by one."""
    shifted = np.roll(np.asarray(factors), 1, axis=1)
    return class_features(scores, train_y, range(shifted.shape[1]), shifted)


def mmices_question_only(image_scores, text_scores, shots, candidate_k=128):
    """Stable text ties preserve the original visual ranking, including fixed questions."""
    image = np.asarray(image_scores)
    text = np.asarray(text_scores)
    if image.shape != text.shape or not shots <= candidate_k <= image.shape[1]:
        raise ValueError("invalid MMICES arrays or candidate budget")
    visual = topk(image, candidate_k)
    selected = np.argsort(-np.take_along_axis(text, visual, axis=1), axis=1, kind="stable")[:, :shots]
    return np.take_along_axis(visual, selected, axis=1)

