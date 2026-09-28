#!/usr/bin/env python
"""Compare learned velocity retrieval with existing DTD runs under fixed parsers."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("results")
SOURCES = {
    "standard": ROOT / "dtd_qwen3vl4b_standard_4shot_full47/seed_73/predictions.jsonl",
    "trajectory": ROOT / "dtd_qwen3vl4b_trajectory_full47/seed_73/predictions.jsonl",
    "papers": ROOT / "dtd_qwen3vl4b_paper_baselines_full47/seed_73/predictions.jsonl",
    "learned": ROOT / "dtd_qwen3vl4b_learned_velocity_full47/seed_73/predictions.jsonl",
}
OUTPUT = ROOT / "dtd_qwen3vl4b_learned_velocity_full47/seed_73/comparison.json"


def compact_label(text: str) -> str:
    """Formatting-only canonical form: lowercase and remove non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", text.strip().lower())


def canonical_prediction(raw: str, parsed: str | None, classes: list[str]) -> str | None:
    # Preserve every valid label already recognized by the original parser.  This is
    # important for DTD, where `dotted` and `polka-dotted` are distinct valid classes.
    if parsed is not None:
        return parsed
    key = compact_label(raw)
    matches = [label for label in classes if compact_label(label) == key]
    return matches[0] if len(matches) == 1 else None


def load_rows() -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for path in SOURCES.values():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            method = row["method"]
            result.setdefault(method, {})[row["query_id"]] = row
    return result


def exact_mcnemar_p(left_wins: int, right_wins: int) -> float:
    n = left_wins + right_wins
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, value) for value in range(min(left_wins, right_wins) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def pairwise(left: np.ndarray, right: np.ndarray, seed: int = 73) -> dict:
    delta = left.astype(np.float32) - right.astype(np.float32)
    left_wins = int(np.sum((left == 1) & (right == 0)))
    right_wins = int(np.sum((left == 0) & (right == 1)))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(20_000, len(delta)))
    boot = delta[draws].mean(axis=1)
    return {
        "difference_percentage_points": float(delta.mean() * 100),
        "left_wins": left_wins,
        "right_wins": right_wins,
        "bootstrap_95_ci_percentage_points": [
            float(np.quantile(boot, 0.025) * 100),
            float(np.quantile(boot, 0.975) * 100),
        ],
        "mcnemar_exact_p": exact_mcnemar_p(left_wins, right_wins),
    }


def main() -> None:
    rows = load_rows()
    learned_name = next(name for name in rows if name.startswith("learned_35_step_velocity_"))
    query_ids = sorted(rows[learned_name])
    classes = sorted({row["target"] for row in rows[learned_name].values()})
    wanted = [
        "class_balanced_random", "rices", "all_layer_velocity", "trajectory_dtw",
        "velocity_dtw", "detriever", "gpt_mm", learned_name,
    ]
    summaries = {}
    arrays = {}
    canonical_arrays = {}
    for method in wanted:
        method_rows = rows[method]
        strict = np.asarray([
            method_rows[query_id]["prediction"] == method_rows[query_id]["target"]
            for query_id in query_ids
        ])
        canonical_predictions = [
            canonical_prediction(
                method_rows[query_id]["raw_output"], method_rows[query_id]["prediction"], classes
            )
            for query_id in query_ids
        ]
        canonical = np.asarray([
            prediction == method_rows[query_id]["target"]
            for query_id, prediction in zip(query_ids, canonical_predictions)
        ])
        rescued = [
            {
                "query_id": query_id,
                "target": method_rows[query_id]["target"],
                "raw_output": method_rows[query_id]["raw_output"],
                "canonical_prediction": prediction,
            }
            for query_id, prediction, before, after in zip(
                query_ids, canonical_predictions, strict, canonical
            )
            if not before and after
        ]
        unresolved = Counter(
            method_rows[query_id]["raw_output"].strip().lower()
            for query_id, prediction in zip(query_ids, canonical_predictions)
            if prediction is None
        )
        summaries[method] = {
            "strict_correct": int(strict.sum()),
            "strict_accuracy": float(strict.mean()),
            "canonical_correct": int(canonical.sum()),
            "canonical_accuracy": float(canonical.mean()),
            "formatting_rescues": rescued,
            "unresolved_invalid_outputs": dict(unresolved),
        }
        arrays[method] = strict
        canonical_arrays[method] = canonical

    comparisons = {}
    for baseline in ["all_layer_velocity", "trajectory_dtw", "detriever", "gpt_mm", "rices"]:
        comparisons[baseline] = {
            "strict": pairwise(arrays[learned_name], arrays[baseline]),
            "canonical": pairwise(canonical_arrays[learned_name], canonical_arrays[baseline]),
        }
    output = {
        "learned_method": learned_name,
        "canonicalization": (
            "Keep any valid label recognized by the original parser; only for parser failures, "
            "lowercase and remove non-alphanumeric characters, then accept an exact unique legal-label match."
        ),
        "summaries": summaries,
        "pairwise_learned_minus_baseline": comparisons,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
