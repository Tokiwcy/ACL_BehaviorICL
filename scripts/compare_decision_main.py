#!/usr/bin/env python
"""Read-only paired comparison of completed Behavior and main-baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def read_run(path: Path) -> tuple[list[tuple[str, str, str]], dict[str, dict[str, bool]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    identity = [(row["sample_id"], row["label"], row["split"])
                for row in manifest["query"]]
    labels = {sample_id: label for sample_id, label, _ in identity}
    if len(labels) != len(identity):
        raise ValueError(f"Duplicate query identity: {path}")
    predictions: dict[str, dict[str, bool]] = {}
    for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        method, query_id = row["method"], row["query_id"]
        if query_id not in labels or row["target"] != labels[query_id]:
            raise ValueError(f"Query/target mismatch: {path} {method} {query_id}")
        if row["correct"] != (row["prediction"] == row["target"]):
            raise ValueError(f"Incorrect correctness flag: {path} {method} {query_id}")
        method_rows = predictions.setdefault(method, {})
        if query_id in method_rows:
            raise ValueError(f"Duplicate prediction: {path} {method} {query_id}")
        method_rows[query_id] = bool(row["correct"])
    return identity, predictions


def paired(left: np.ndarray, right: np.ndarray, bootstrap: int,
           generator: np.random.Generator) -> dict:
    wins = int(((left == 1) & (right == 0)).sum())
    losses = int(((left == 0) & (right == 1)).sum())
    delta = left.astype(np.int8) - right.astype(np.int8)
    samples = np.empty(bootstrap, dtype=np.float64)
    for start in range(0, bootstrap, 500):
        stop = min(start + 500, bootstrap)
        indices = generator.integers(0, len(delta), size=(stop - start, len(delta)))
        samples[start:stop] = delta[indices].mean(axis=1)
    discordant = wins + losses
    p_value = (float(binomtest(min(wins, losses), discordant, 0.5).pvalue)
               if discordant else 1.0)
    return {
        "left_only_correct": wins,
        "right_only_correct": losses,
        "difference_pp": 100.0 * float(delta.mean()),
        "bootstrap_95pct_ci_pp": [100.0 * float(x)
                                   for x in np.quantile(samples, [0.025, 0.975])],
        "exact_mcnemar_p": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-run", type=Path, required=True)
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--detriever-run", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if args.bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    reference = None
    all_rows: dict[str, dict[str, bool]] = {}
    for path in (args.decision_run, args.main_run, args.detriever_run):
        identity, predictions = read_run(path)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"Query ID/label/split/order mismatch: {path}")
        for method, rows in predictions.items():
            if method in all_rows:
                raise ValueError(f"Duplicate method across runs: {method}")
            all_rows[method] = rows
    assert reference is not None
    order = ("rices", "gpt_mm", "detriever", "decision_zero", "decision_learn")
    ids = [sample_id for sample_id, _, _ in reference]
    expected = set(ids)
    vectors: dict[str, np.ndarray] = {}
    for method in order:
        rows = all_rows.get(method)
        if rows is None or set(rows) != expected:
            raise ValueError(f"Missing/incomplete method: {method}")
        vectors[method] = np.array([rows[sample_id] for sample_id in ids], dtype=np.int8)
    generator = np.random.default_rng(args.seed)
    comparisons = {}
    for left, right in (("decision_learn", "decision_zero"),
                        ("decision_learn", "detriever"),
                        ("decision_learn", "gpt_mm"),
                        ("decision_learn", "rices")):
        comparisons[f"{left}__vs__{right}"] = paired(
            vectors[left], vectors[right], args.bootstrap, generator)
    result = {
        "queries": len(reference),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "accuracy": {method: {"correct": int(vector.sum()), "total": len(vector),
                              "accuracy": float(vector.mean())}
                     for method, vector in vectors.items()},
        "pairwise": comparisons,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
