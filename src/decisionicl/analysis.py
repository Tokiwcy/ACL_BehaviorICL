"""Paired answers, selector diagnostics and separate repeated-run summaries."""
from __future__ import annotations
import numpy as np
from .metrics import accuracy_metrics, paired_comparison


def evaluate(labels, baseline, method, *, class_names, bootstrap_samples=10000, seed=42):
    result = {}
    for name, predictions in (("baseline", baseline), ("method", method)):
        result[name] = {**accuracy_metrics(labels, predictions),
                        "invalid": sum(x not in class_names for x in predictions)}
    result["paired"] = paired_comparison(labels=labels, baseline_predictions=baseline,
        method_predictions=method, bootstrap_samples=bootstrap_samples, seed=seed)
    return result


def repeated_summary(per_run):
    """Rows are runs, columns are cells; equal-cell mean is formed within each run."""
    values = np.asarray(per_run, np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("expected at least two complete finite run vectors")
    means = values.mean(axis=1)
    return {"cell_mean": values.mean(0).tolist(), "cell_sample_sd": values.std(0, ddof=1).tolist(),
            "equal_cell_run_means": means.tolist(), "equal_cell_mean": float(means.mean()),
            "equal_cell_sample_sd": float(means.std(ddof=1)), "runs": len(values)}


def plan_changes(baseline_rows, method_rows):
    a = {row["query_id"]: row for row in baseline_rows}
    b = {row["query_id"]: row for row in method_rows}
    if len(a) != len(baseline_rows) or len(b) != len(method_rows) or a.keys() != b.keys():
        raise ValueError("plans require identical unique query coverage")
    changed = sum(a[q]["demo_ids"] != b[q]["demo_ids"] for q in a)
    return {"queries": len(a), "ordered_demo_changed": changed,
            "ordered_demo_identical": len(a)-changed}

