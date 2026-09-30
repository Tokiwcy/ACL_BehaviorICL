#!/usr/bin/env python
"""Analyze paired accuracy and retrieval diagnostics for one pilot run."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def exact_mcnemar_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--additional-run-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads((args.run_dir / "run_identity.json").read_text(encoding="utf-8"))["dataset"]
    selections = json.loads((args.run_dir / "selections.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    for additional_dir in args.additional_run_dirs:
        additional_manifest = json.loads(
            (additional_dir / "manifest.json").read_text(encoding="utf-8")
        )
        # Independently run cloud tuples can have different Windows/Linux image
        # paths. Pair only after checking every official sample ID, label and split
        # in the exact original order; image paths are transport metadata.
        for split in ("bank", "query"):
            original = [(row["sample_id"], row["label"], row["split"])
                        for row in manifest[split]]
            additional = [(row["sample_id"], row["label"], row["split"])
                          for row in additional_manifest[split]]
            if additional != original:
                raise ValueError(f"{split} identity/order mismatch in additional run: {additional_dir}")
        additional_selections = json.loads(
            (additional_dir / "selections.json").read_text(encoding="utf-8")
        )
        overlap = set(selections) & set(additional_selections)
        if overlap:
            raise ValueError(f"Duplicate methods across runs: {sorted(overlap)}")
        selections.update(additional_selections)
        rows.extend(
            json.loads(line)
            for line in (additional_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
    query = {sample["sample_id"]: sample for sample in manifest["query"]}
    methods = args.methods or list(selections)
    by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][row["query_id"]] = row

    missing = {
        method: sorted(set(query) - set(by_method[method]))
        for method in methods
        if set(query) - set(by_method[method])
    }
    if missing:
        counts = ", ".join(f"{method}: {len(ids)} missing" for method, ids in missing.items())
        raise ValueError(f"Cannot run paired analysis with incomplete methods ({counts})")

    base = "rices"
    query_ids = list(query)
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, len(query_ids), size=(args.bootstrap, len(query_ids)))
    analysis: dict[str, dict] = {}
    base_sets = {
        query_id: {sample["sample_id"] for sample in selections[base][query_id]}
        for query_id in query_ids
    }
    base_correct = np.asarray([by_method[base][query_id]["correct"] for query_id in query_ids], dtype=float)

    for method in methods:
        correct = np.asarray([by_method[method][query_id]["correct"] for query_id in query_ids], dtype=float)
        per_class = {}
        for label in sorted({sample["label"] for sample in query.values()}):
            indices = [i for i, query_id in enumerate(query_ids) if query[query_id]["label"] == label]
            per_class[label] = {
                "correct": int(correct[indices].sum()),
                "total": len(indices),
                "accuracy": float(correct[indices].mean()),
            }

        same_label_fractions = []
        jaccards = []
        exact_sets = 0
        for query_id in query_ids:
            demos = selections[method][query_id]
            same_label_fractions.append(
                sum(sample["label"] == query[query_id]["label"] for sample in demos) / len(demos)
            )
            selected = {sample["sample_id"] for sample in demos}
            reference = base_sets[query_id]
            jaccards.append(len(selected & reference) / len(selected | reference))
            exact_sets += selected == reference

        wins = int(np.sum((correct == 1) & (base_correct == 0)))
        losses = int(np.sum((correct == 0) & (base_correct == 1)))
        paired_diff = (correct[sampled] - base_correct[sampled]).mean(axis=1)
        analysis[method] = {
            "correct": int(correct.sum()),
            "total": len(query_ids),
            "accuracy": float(correct.mean()),
            "parse_failures": sum(
                by_method[method][query_id]["prediction"] is None for query_id in query_ids
            ),
            "per_class": per_class,
            "same_label_demo_fraction": float(np.mean(same_label_fractions)),
            "mean_jaccard_vs_rices": float(np.mean(jaccards)),
            "exact_same_set_as_rices": exact_sets,
            "paired_vs_rices": {
                "wins": wins,
                "losses": losses,
                "net_correct": wins - losses,
                "accuracy_difference": float(correct.mean() - base_correct.mean()),
                "bootstrap_95pct_ci": [float(x) for x in np.quantile(paired_diff, [0.025, 0.975])],
                "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
            },
        }

    pairwise = {}
    for left, right in combinations(methods, 2):
        left_correct = np.asarray(
            [by_method[left][query_id]["correct"] for query_id in query_ids], dtype=float
        )
        right_correct = np.asarray(
            [by_method[right][query_id]["correct"] for query_id in query_ids], dtype=float
        )
        wins = int(np.sum((left_correct == 1) & (right_correct == 0)))
        losses = int(np.sum((left_correct == 0) & (right_correct == 1)))
        paired_diff = (left_correct[sampled] - right_correct[sampled]).mean(axis=1)
        pairwise[f"{left}__vs__{right}"] = {
            "left": left,
            "right": right,
            "wins": wins,
            "losses": losses,
            "accuracy_difference": float(left_correct.mean() - right_correct.mean()),
            "bootstrap_95pct_ci": [
                float(x) for x in np.quantile(paired_diff, [0.025, 0.975])
            ],
            "exact_mcnemar_p": exact_mcnemar_p(wins, losses),
        }

    (args.run_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.run_dir / "pairwise.json").write_text(
        json.dumps(pairwise, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        method: {
            key: analysis[method][key]
            for key in ("correct", "total", "accuracy", "parse_failures")
        }
        for method in methods
    }
    # A cross-run analysis combines methods from independent tuples; do not
    # replace the primary run's own completion summary with that merged view.
    summary_name = "paired_summary.json" if args.additional_run_dirs else "summary.json"
    (args.run_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Full {dataset} hidden-state retrieval analysis",
        "",
        f"Queries: {len(query_ids)}. RICES is the paired reference.",
        "",
        "| Method | Accuracy | Same-label demos | Jaccard vs RICES | Wins / losses vs RICES | Paired 95% bootstrap CI | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        item = analysis[method]
        paired = item["paired_vs_rices"]
        ci = paired["bootstrap_95pct_ci"]
        lines.append(
            f"| {method} | {item['correct']}/{item['total']} ({item['accuracy']:.2%}) "
            f"| {item['same_label_demo_fraction']:.3f} | {item['mean_jaccard_vs_rices']:.3f} "
            f"| {paired['wins']} / {paired['losses']} | [{ci[0]:.1%}, {ci[1]:.1%}] "
            f"| {paired['exact_mcnemar_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise hidden-state comparisons",
            "",
            "| Left - right | Accuracy difference | Wins / losses | Paired 95% bootstrap CI | McNemar p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    hidden_methods = [method for method in methods if method not in {"class_balanced_random", "rices"}]
    for left, right in combinations(hidden_methods, 2):
        item = pairwise[f"{left}__vs__{right}"]
        ci = item["bootstrap_95pct_ci"]
        lines.append(
            f"| {left} - {right} | {item['accuracy_difference']:.1%} "
            f"| {item['wins']} / {item['losses']} | [{ci[0]:.1%}, {ci[1]:.1%}] "
            f"| {item['exact_mcnemar_p']:.4f} |"
        )
    lines.extend(["", "## Per-class accuracy", ""])
    labels = list(next(iter(analysis.values()))["per_class"])
    lines.append("| Method | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---:|" * len(labels))
    for method in methods:
        values = [analysis[method]["per_class"][label] for label in labels]
        lines.append("| " + method + " | " + " | ".join(f"{x['correct']}/{x['total']}" for x in values) + " |")
    lines.extend(
        [
            "",
            "The confidence interval and McNemar test are paired over the same queries. They measure uncertainty from this query sample only; they do not account for candidate-bank or seed variation.",
        ]
    )
    (args.run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
