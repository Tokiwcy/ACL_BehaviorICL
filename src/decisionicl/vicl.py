"""VICL text-to-image retrieval with visual and index tie breaks."""
import numpy as np

def stable_top1(primary, secondary=None, candidates=None):
    primary = np.asarray(primary)
    if candidates is None:
        candidates = np.arange(primary.shape[0], dtype=np.int64)
    else:
        candidates = np.asarray(candidates, np.int64)
    if secondary is None:
        secondary = np.zeros_like(primary)
    order = np.lexsort((
        candidates,
        -np.asarray(secondary)[candidates],
        -primary[candidates]))
    return int(candidates[order[0]])


def top2(primary, secondary=None, candidates=None):
    scores = np.asarray(primary)
    ids = np.arange(len(scores), dtype=np.int64) if candidates is None else np.asarray(candidates, np.int64)
    other = np.zeros_like(scores) if secondary is None else np.asarray(secondary)
    if len(ids) < 2 or len(set(ids.tolist())) != len(ids):
        raise ValueError('two distinct candidates required')
    order = np.lexsort((ids, -other[ids], -scores[ids]))
    return ids[order[:2]].tolist()
