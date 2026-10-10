"""Verify every shared 32-image query state matches independent TAM prefill capture."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT / "analysis/query_retrieval_local_v1/protocol.json").read_text(encoding="utf-8"))
    compared = 0
    for dataset, table in plan["datasets"].items():
        bank_n = len(table["bank"])
        for arm in ("generic", "task_short"):
            values = np.load(ROOT / "results/query_retrieval_local_v1" / dataset / f"{arm}_states.npy", mmap_mode="r")
            for i, sample in enumerate(table["query"]):
                path = ROOT / "results/query_tam_local_v1" / f"{dataset}__{sample['sample_id']}__{arm}__fixed/maps.npz"
                tam = np.load(path, allow_pickle=False)["anchor_states"]
                if not np.array_equal(values[bank_n + i], tam.astype(np.float16)):
                    raise ValueError(f"TAM/retrieval anchor mismatch: {dataset}/{sample['sample_id']}/{arm}")
                compared += 1
    print(json.dumps({"exact_float16_anchor_matches": compared, "datasets": 5, "query_images": 160}))


if __name__ == "__main__":
    main()
