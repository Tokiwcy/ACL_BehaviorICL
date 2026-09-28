#!/usr/bin/env python
"""Canonical full-split manifests for the five fine-grained ICL benchmarks.

The main experiment contract is intentionally explicit: every dataset/model/seed run
gets a freshly initialized learned retriever.  Checkpoints are scoped by all three keys
and may never be silently reused across them.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import scipy.io


DATASETS = ("dtd", "aircraft", "cub", "dogs", "pets")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: str
    label: str
    split: str


def _sample(sample_id: str, path: Path, label: str, split: str) -> Sample:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return Sample(sample_id, str(resolved), label, split)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dtd(root: Path) -> tuple[list[Sample], list[Sample]]:
    def read(names: list[str], split: str) -> list[Sample]:
        rows = []
        for name in names:
            for rel in _read_lines(root / "labels" / name):
                label = rel.split("/", 1)[0]
                rows.append(_sample(Path(rel).stem, root / "images" / rel, label, split))
        return rows

    return read(["train1.txt", "val1.txt"], "bank"), read(["test1.txt"], "query")


def load_aircraft(root: Path) -> tuple[list[Sample], list[Sample]]:
    data = root / "data"

    def read(name: str, split: str) -> list[Sample]:
        rows = []
        for line in _read_lines(data / name):
            image_id, label = line.split(" ", 1)
            rows.append(_sample(image_id, data / "images" / f"{image_id}.jpg", label, split))
        return rows

    return read("images_variant_trainval.txt", "bank"), read("images_variant_test.txt", "query")


def _pretty_cub_label(value: str) -> str:
    value = re.sub(r"^\d+\.", "", value)
    return value.replace("_", " ").replace("-", " ").lower()


def load_cub(root: Path) -> tuple[list[Sample], list[Sample]]:
    images = {
        int(index): rel for index, rel in (line.split(" ", 1) for line in _read_lines(root / "images.txt"))
    }
    image_labels = {
        int(index): int(label)
        for index, label in (line.split() for line in _read_lines(root / "image_class_labels.txt"))
    }
    class_labels = {
        int(index): _pretty_cub_label(label)
        for index, label in (line.split(" ", 1) for line in _read_lines(root / "classes.txt"))
    }
    train_flags = {
        int(index): int(flag)
        for index, flag in (line.split() for line in _read_lines(root / "train_test_split.txt"))
    }
    bank, query = [], []
    for index in sorted(images):
        split = "bank" if train_flags[index] else "query"
        row = _sample(
            f"cub_{index:05d}", root / "images" / images[index], class_labels[image_labels[index]], split
        )
        (bank if split == "bank" else query).append(row)
    return bank, query


def _mat_string(value) -> str:
    while hasattr(value, "shape") and value.size == 1 and not isinstance(value, str):
        value = value.flat[0]
    return str(value)


def _pretty_dog_label(rel: str) -> str:
    folder = rel.replace("\\", "/").split("/", 1)[0]
    label = folder.split("-", 1)[1]
    return label.replace("_", " ").replace("-", " ").lower()


def load_dogs(root: Path) -> tuple[list[Sample], list[Sample]]:
    def read(name: str, split: str) -> list[Sample]:
        data = scipy.io.loadmat(root / name)
        rows = []
        for value in data["file_list"].flat:
            rel = _mat_string(value).replace("\\", "/")
            rows.append(
                _sample(f"dogs_{Path(rel).stem}", root / "Images" / rel, _pretty_dog_label(rel), split)
            )
        return rows

    return read("train_list.mat", "bank"), read("test_list.mat", "query")


def _pretty_pet_label(image_id: str) -> str:
    return re.sub(r"_\d+$", "", image_id).replace("_", " ").lower()


def load_pets(root: Path) -> tuple[list[Sample], list[Sample]]:
    def read(name: str, split: str) -> list[Sample]:
        rows = []
        for line in _read_lines(root / "annotations" / name):
            image_id = line.split()[0]
            rows.append(
                _sample(f"pets_{image_id}", root / "images" / f"{image_id}.jpg", _pretty_pet_label(image_id), split)
            )
        return rows

    return read("trainval.txt", "bank"), read("test.txt", "query")


LOADERS = {
    "dtd": (load_dtd, "dtd"),
    "aircraft": (load_aircraft, "fgvc-aircraft-2013b"),
    "cub": (load_cub, "CUB_200_2011"),
    "dogs": (load_dogs, "stanford_dogs"),
    "pets": (load_pets, "oxford_iiit_pet"),
}


def load_dataset(name: str, datasets_root: Path = Path("datasets")) -> tuple[list[Sample], list[Sample]]:
    loader, dirname = LOADERS[name]
    bank, query = loader(datasets_root / dirname)
    ids = [sample.sample_id for sample in bank + query]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample ids in {name}")
    bank_labels = {sample.label for sample in bank}
    query_labels = {sample.label for sample in query}
    if bank_labels != query_labels:
        raise ValueError(f"Bank/query label spaces disagree in {name}")
    return bank, query


def run_directory(output_root: Path, dataset: str, model_slug: str, seed: int) -> Path:
    """Scope every trained checkpoint so cross-run reuse cannot happen accidentally."""
    return output_root / dataset / model_slug / f"seed_{seed}"


def protocol_metadata(dataset: str, model: str, seed: int, bank: list[Sample], query: list[Sample]) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "bank_count": len(bank),
        "query_count": len(query),
        "class_count": len({sample.label for sample in bank}),
        "split": "official full train/trainval bank and official full test query",
        "training_contract": {
            "cdr_zero": "no training",
            "cdr_learn": "fresh random initialization for this dataset/model/seed",
            "detriever": "fresh random initialization for this dataset/model/seed",
            "gpt_mm": "no retriever training",
            "rices": "no retriever training",
            "cross_dataset_or_model_checkpoint_reuse": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/multidataset_manifests"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    summary = {}
    for name in args.datasets:
        bank, query = load_dataset(name, args.datasets_root)
        target = args.output_dir / name
        target.mkdir(parents=True, exist_ok=True)
        metadata = protocol_metadata(name, args.model, args.seed, bank, query)
        (target / "manifest.json").write_text(
            json.dumps({"metadata": metadata, "bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[name] = metadata
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
