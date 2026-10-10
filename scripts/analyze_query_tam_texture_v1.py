"""Descriptive DTD map spread without inventing an object-region ground truth."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def entropy(values):
    x = np.maximum(np.asarray(values, dtype=float).ravel(), 0)
    if x.sum() <= 0:
        return None
    p = x / x.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(x)))


def main():
    plan = json.loads((ROOT / "analysis/query_tam_local_v3/protocol.json").read_text(encoding="utf-8"))
    rows = []
    for example in plan["examples"]:
        if example["dataset"] != "dtd":
            continue
        entropies = {}
        for arm in ("generic", "task_short"):
            folder = ROOT / "results/query_tam_local_v1" / f"dtd__{example['sample_id']}__{arm}__fixed"
            values = np.load(folder / "maps.npz", allow_pickle=False)["maps"][0]
            entropies[arm] = entropy(values)
        rows.append({"sample_id": example["sample_id"], **entropies,
                     "delta": entropies["task_short"] - entropies["generic"]})
    delta = np.array([row["delta"] for row in rows])
    summary = {"n": len(rows), "generic_mean": float(np.mean([x["generic"] for x in rows])),
               "task_mean": float(np.mean([x["task_short"] for x in rows])),
               "paired_delta_mean": float(delta.mean()), "paired_delta_median": float(np.median(delta)),
               "positive": int((delta > 0).sum()), "negative": int((delta < 0).sum()),
               "interpretation": "Normalized spatial entropy is descriptive. DTD has no independent ground-truth region; higher spread is not known to be more correct."}
    out = ROOT / "analysis/query_tam_local_v3/texture_spread.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
