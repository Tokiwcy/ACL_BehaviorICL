"""A compact array-level CLI. Discovery/encoding/model APIs are documented separately."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .pipeline import DecisionICLRetriever
from .schema import Sample
from .analysis import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="decisionicl")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "retrieve"):
        sub = commands.add_parser(name)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--contract", type=Path, required=True)
        sub.add_argument("--shots", type=int, required=True)
        sub.add_argument("--calibration", type=Path, required=True)
        if name == "retrieve":
            sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("evaluate")
    sub.add_argument("--input", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        data = json.loads(args.input.read_text(encoding="utf-8"))
        result = evaluate(data["labels"], data["baseline"], data["method"], class_names=data["class_names"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return
    with np.load(args.input, allow_pickle=False) as bundle:
        # NPZ contains aligned full-pool parent scores and one probability matrix per factor.
        count = int(bundle["factor_count"])
        train_probabilities = [bundle[f"train_probabilities_{i}"] for i in range(count)]
        query_probabilities = [bundle[f"query_probabilities_{i}"] for i in range(count)]
        retriever = DecisionICLRetriever(bundle["train_ids"].tolist(), bundle["train_labels"].tolist(),
            train_probabilities, contract=json.loads(args.contract.read_text(encoding="utf-8")), shots=args.shots)
        ids = bundle["query_ids"].tolist()
        if args.command == "fit":
            if str(bundle["query_split"]) != "validation":
                parser.error("fit requires query_split=validation")
            samples = [Sample(sample_id=qid, image_path="", label=label, split="validation")
                       for qid, label in zip(ids, bundle["query_labels"].tolist())]
            result = retriever.fit(bundle["parent_scores"], query_probabilities, samples)
            args.calibration.parent.mkdir(parents=True, exist_ok=True)
            args.calibration.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            if str(bundle["query_split"]) != "test":
                parser.error("retrieve requires query_split=test")
            retriever.load_calibration(json.loads(args.calibration.read_text(encoding="utf-8")))
            result = retriever.retrieve(bundle["parent_scores"], query_probabilities, ids)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(row)+"\n" for row in result), encoding="utf-8")

