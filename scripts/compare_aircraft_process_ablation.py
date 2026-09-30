#!/usr/bin/env python
"""Read-only, same-query paired comparison for Aircraft process ablations."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


def read_run(path: Path) -> tuple[list[tuple[str, str, str]], dict[str, dict[str, bool]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    identity = [(item["sample_id"], item["label"], item["split"])
                for item in manifest["query"]]
    labels = {sample_id: label for sample_id, label, _ in identity}
    predictions: dict[str, dict[str, bool]] = {}
    for line in (path / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        method, query_id = row["method"], row["query_id"]
        if query_id not in labels or row["target"] != labels[query_id]:
            raise ValueError(f"Query/label mismatch: {path} {method} {query_id}")
        if row["correct"] != (row["prediction"] == labels[query_id]):
            raise ValueError(f"Correctness mismatch: {path} {method} {query_id}")
        method_rows = predictions.setdefault(method, {})
        if query_id in method_rows:
            raise ValueError(f"Duplicate prediction: {path} {method} {query_id}")
        method_rows[query_id] = bool(row["correct"])
    return identity, predictions


def exact_mcnemar_p(wins: int, losses: int) -> float:
    n = wins + losses
    if not n:
        return 1.0
    lower_tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run", type=Path, required=True)
    parser.add_argument("--decision-run", type=Path, required=True)
    parser.add_argument("--forward-run", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    runs = [args.main_run, args.decision_run]
    if args.forward_run:
        runs.append(args.forward_run)
    reference: list[tuple[str, str, str]] | None = None
    all_rows: dict[str, dict[str, bool]] = {}
    for path in runs:
        identity, predictions = read_run(path)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"Aircraft query ID/label/split/order mismatch: {path}")
        for method, rows in predictions.items():
            if method in all_rows and method in {
                "cdr_zero", "cdr_learn", "decision_zero", "decision_learn",
                "forward_zero", "forward_learn",
            }:
                raise ValueError(f"Duplicate method across runs: {method}")
            all_rows[method] = rows
    assert reference is not None
    order = [name for name in ("cdr_zero", "cdr_learn", "decision_zero",
                               "decision_learn", "forward_zero", "forward_learn")
             if name in all_rows]
    expected = set(sample_id for sample_id, _, _ in reference)
    vectors: dict[str, np.ndarray] = {}
    for method in order:
        if set(all_rows[method]) != expected:
            raise ValueError(f"Incomplete method {method}: {len(all_rows[method])}/{len(expected)}")
        vectors[method] = np.array([all_rows[method][sample_id]
                                    for sample_id, _, _ in reference], dtype=np.int8)

    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, len(reference), size=(args.bootstrap, len(reference)))
    comparisons = {}
    for left, right in combinations(order, 2):
        lhs, rhs = vectors[left], vectors[right]
        wins = int(((lhs == 1) & (rhs == 0)).sum())
        losses = int(((lhs == 0) & (rhs == 1)).sum())
        difference = lhs.astype(np.int8) - rhs.astype(np.int8)
        distribution = difference[sampled].mean(axis=1)
        comparisons[f"{left}__vs__{right}"] = {
            "wins": wins,
            "losses": losses,
            "difference_pp": 100 * float(difference.mean()),
            "bootstrap_95pct_ci_pp": [100 * float(value)
                                       for value in np.quantile(distribution, [0.025, 0.975])],
            "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
        }
    output = {
        "queries": len(reference),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "accuracy": {method: {"correct": int(vector.sum()),
                              "total": len(reference),
                              "accuracy": float(vector.mean())}
                     for method, vector in vectors.items()},
        "pairwise": comparisons,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
