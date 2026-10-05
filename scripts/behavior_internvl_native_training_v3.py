"""Exact historical numerical helpers; no Qwen adapter/model import."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityRetriever(nn.Module):
    """Layer-weighted cosine kernel, optionally with one shared low-rank projection."""

    def __init__(self, width: int, layer_count: int, projection_dim: int | None) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.projection = (
            nn.Linear(width, projection_dim, bias=False) if projection_dim is not None else None
        )
        if self.projection is not None:
            nn.init.orthogonal_(self.projection.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        velocity = states[:, 1:] - states[:, :-1]
        velocity = F.normalize(velocity.float(), dim=-1)
        if self.projection is not None:
            velocity = self.projection(velocity)
        return F.normalize(velocity, dim=-1)

    def weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(dim=0)

    def score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ald,bld,l->ab", left, right, self.weights())


class StateRetriever(VelocityRetriever):
    """Parameter-matched control that scores normalized raw layer states."""

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = F.normalize(states.float(), dim=-1)
        if self.projection is not None:
            encoded = self.projection(encoded)
        return F.normalize(encoded, dim=-1)


def supervised_contrastive_loss(
    scores: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Training scores must be a square pairwise matrix")
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~diagonal
    valid = ~diagonal
    if not positives.any(dim=1).all():
        raise ValueError("Every anchor needs at least one positive")
    logits = scores / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    negative_inf = torch.finfo(logits.dtype).min
    numerator = torch.logsumexp(logits.masked_fill(~positives, negative_inf), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~valid, negative_inf), dim=1)
    return -(numerator - denominator).mean()


def balanced_batch(
    by_label: dict[int, np.ndarray], classes_per_batch: int, samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(sorted(by_label))
    chosen = rng.choice(keys, size=classes_per_batch, replace=classes_per_batch > len(keys))
    indices, labels = [], []
    for label in chosen:
        pool = by_label[int(label)]
        selected = rng.choice(pool, size=samples_per_class, replace=samples_per_class > len(pool))
        indices.extend(selected.tolist())
        labels.extend([int(label)] * samples_per_class)
    return np.asarray(indices), np.asarray(labels)


@torch.inference_mode()
def encode_indices(
    model: VelocityRetriever, states: np.ndarray, indices: np.ndarray, chunk_size: int = 64
) -> torch.Tensor:
    model.eval()
    encoded = []
    for start in range(0, len(indices), chunk_size):
        batch = torch.from_numpy(
            np.asarray(states[indices[start : start + chunk_size]], dtype=np.float32)
        ).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded.append(model(batch).float())
        del batch
    return torch.cat(encoded)


@torch.inference_mode()
def retrieval_metrics(
    model: VelocityRetriever,
    states: np.ndarray,
    candidate_indices: np.ndarray,
    query_indices: np.ndarray,
    labels: list[str],
    shots: int,
) -> dict:
    candidate = encode_indices(model, states, candidate_indices)
    query = encode_indices(model, states, query_indices)
    scores = model.score(query, candidate)
    top = scores.topk(shots, dim=1).indices.cpu().numpy()
    candidate_labels = np.asarray(labels)[candidate_indices]
    query_labels = np.asarray(labels)[query_indices]
    retrieved = candidate_labels[top]
    matches = retrieved == query_labels[:, None]
    reciprocal = []
    full_order = scores.argsort(dim=1, descending=True).cpu().numpy()
    for row, label in zip(full_order, query_labels):
        ranks = np.flatnonzero(candidate_labels[row] == label)
        reciprocal.append(1.0 / (int(ranks[0]) + 1))
    return {
        "top4_same_label_fraction": float(matches.mean()),
        "top4_any_same_label": float(matches.any(axis=1).mean()),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
    }

