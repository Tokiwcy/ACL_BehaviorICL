"""Contract-bound calibration and offline plan construction for the main method."""
from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .hashing import stable_hash, array_sha256
from .folds import stratified_folds
from .selection import (
    calibrate_decisionicl, class_factor_prototypes, factor_class_similarities,
    class_features, select_decisionicl_demonstrations,
)


class DecisionICLRetriever:
    """One exact target, dataset, bank, parent, and shot budget per fitted instance.

    Query labels are deliberately absent from retrieve(). Inputs must already be
    aligned to train_ids/validation samples by the caller; this class binds their
    order and numeric identities into the saved calibration.
    """
    def __init__(self, train_ids, train_labels, train_probabilities, *, contract, shots):
        if len(train_ids) != len(set(train_ids)) or len(train_ids) != len(train_labels):
            raise ValueError("training IDs must be unique and align with labels")
        required = {"target_model", "dataset", "bank_sha256", "evidence_encoder", "parent", "seed"}
        if not required <= contract.keys():
            raise ValueError(f"missing method identity: {sorted(required-contract.keys())}")
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be positive")
        self.train_ids = tuple(train_ids)
        self.train_labels = tuple(train_labels)
        self.classes, self.prototypes = class_factor_prototypes(train_labels, train_probabilities)
        self.shots = shots
        self.contract = {**contract, "shots": shots, "train_ids": self.train_ids,
                         "train_labels": self.train_labels,
                         "train_evidence_sha256": [array_sha256(np.asarray(p)) for p in train_probabilities],
                         "selection": "class-prototype-jsd-equal-factor-full-pool-v1"}
        self.contract_hash = stable_hash(self.contract)
        self.calibration = None

    def features(self, parent_scores, probabilities):
        factors = factor_class_similarities(probabilities, self.prototypes)
        return class_features(parent_scores, self.train_labels, self.classes, factors)

    def fit(self, parent_scores, probabilities, validation_samples):
        if not validation_samples or any(s.split != "validation" for s in validation_samples):
            raise ValueError("fit requires validation samples only")
        ids = [s.sample_id for s in validation_samples]
        if len(ids) != len(set(ids)) or set(ids) & set(self.train_ids):
            raise ValueError("validation IDs duplicate or overlap train")
        if any(s.label not in self.classes for s in validation_samples):
            raise ValueError("validation has an unknown class")
        features = self.features(parent_scores, probabilities)
        folds = stratified_folds(validation_samples, 5, 42)
        fitted = calibrate_decisionicl(features, [self.classes.index(s.label) for s in validation_samples], folds)
        self.calibration = {
            "contract_hash": self.contract_hash, "contract": self.contract,
            "validation_ids": ids, "validation_features_sha256": array_sha256(features),
            "validation_labels_sha256": stable_hash([s.label for s in validation_samples]),
            "folds": folds.tolist(), "test_used": False,
            **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(fitted).items()},
        }
        return self.calibration

    def load_calibration(self, calibration):
        if calibration["contract_hash"] != self.contract_hash or calibration["test_used"] is not False:
            raise ValueError("calibration belongs to a different method/data/budget contract")
        if stable_hash(calibration["contract"]) != self.contract_hash:
            raise ValueError("corrupt calibration contract")
        weights = np.asarray(calibration["weights"])
        if (weights.shape != (len(self.prototypes)+1,) or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
            raise ValueError("invalid calibration weights")
        self.calibration = calibration

    def retrieve(self, parent_scores, query_probabilities, query_ids):
        if self.calibration is None:
            raise ValueError("fit or load a matching calibration first")
        ids = tuple(query_ids)
        if len(ids) != len(set(ids)) or set(ids) & (set(self.train_ids) | set(self.calibration["validation_ids"])):
            raise ValueError("test IDs duplicate or overlap calibration data")
        features = self.features(parent_scores, query_probabilities)
        if features.shape[0] != len(ids):
            raise ValueError("query IDs and scores do not align")
        weights = np.asarray(self.calibration["weights"], np.float32)
        demos = select_decisionicl_demonstrations(parent_scores, self.train_ids, self.train_labels,
                                                 self.classes, features, weights, self.shots)
        selected = np.einsum("qcf,f->qc", features, weights).argmax(1)
        return [{"query_id": qid, "demo_ids": list(chosen), "selected_class": self.classes[int(cls)],
                 "contract_hash": self.contract_hash, "test_label_used": False}
                for qid, chosen, cls in zip(ids, demos, selected)]

