"""Generic direct-sample control: distinct objective and retrieval structure."""
from __future__ import annotations
import numpy as np
from .evidence import EvidenceStore, group_similarity_matrix
from .folds import stratified_folds
from .selection import ALPHA_GRID as ALPHAS

def scaled_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.clip(high - low, 1e-8, None)


def atom_scores(query: EvidenceStore, train: EvidenceStore) -> np.ndarray:
    if query.group_ids != train.group_ids or not query.group_ids:
        raise ValueError("Generic direct evidence group contract drift")
    result = np.zeros((len(query.sample_ids), len(train.sample_ids)), np.float32)
    for group_index in range(len(query.group_ids)):
        result += group_similarity_matrix(
            query, train, group_index, validate_alignment=(group_index == 0))
    return result / float(len(query.group_ids))


def combined(base_scores: np.ndarray, generic_scores: np.ndarray, alpha: float) -> np.ndarray:
    return float(alpha) * scaled_rows(base_scores) + (1.0 - float(alpha)) * generic_scores


def top(scores: np.ndarray, shots: int) -> np.ndarray:
    return np.argsort(-scores, axis=1, kind="stable")[:, :shots]


def quality(picks: np.ndarray, truth: np.ndarray, train_y: np.ndarray) -> float:
    return float(np.mean(train_y[picks] == truth[:, None]))


def fit_one(method: str, shots: int, bases: tuple[np.ndarray, ...], generic: np.ndarray,
            validation_y: np.ndarray, train_y: np.ndarray, validation) -> dict:
    candidates = {}
    for alpha in ALPHAS:
        selected = [top(combined(values, generic, alpha), 1 if len(bases) == 2 else shots)
                    for values in bases]
        candidates[alpha] = (np.concatenate(selected, axis=1)
                             if len(selected) == 2 else selected[0])
    folds = stratified_folds(validation, 5, 42)
    oof = np.empty((len(validation), shots), np.int64)
    history = []
    for fold in range(5):
        fit = np.flatnonzero(folds != fold)
        hold = np.flatnonzero(folds == fold)
        scored = {a: quality(p[fit], validation_y[fit], train_y) for a, p in candidates.items()}
        alpha = max(ALPHAS, key=lambda a: (scored[a], a))
        oof[hold] = candidates[alpha][hold]
        history.append(dict(fold=fold, alpha=alpha, fit_demo_same_class_precision=scored[alpha],
                            fit_count=len(fit), heldout_count=len(hold)))
    scored = {a: quality(p, validation_y, train_y) for a, p in candidates.items()}
    alpha = max(ALPHAS, key=lambda a: (scored[a], a))
    return dict(alpha=alpha, alpha_grid_scores={str(a): scored[a] for a in ALPHAS},
                outer_folds=history, oof_demo_same_class_precision=quality(oof, validation_y, train_y),
                full_fit_demo_same_class_precision=scored[alpha], independently_fit=True,
                inherited_weights=False, objective="validation demo same-class precision",
                direct_candidate_retrieval=True, class_prototype=False,
                group_weights="equal fixed mean", parent_mass_tie_rule="prefer larger alpha")
