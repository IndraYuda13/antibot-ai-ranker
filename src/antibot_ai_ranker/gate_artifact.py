from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .balanced_validation import split_manual_calibration
from .dataset import Example
from .disagreement_gate import _predict, _rows_from_examples
from .override import train_override_classifier
from .train import train_perceptron

SCHEMA_VERSION = "antibot-ai-ranker.disagreement-gate.v1"


@dataclass(frozen=True)
class GateArtifact:
    schema_version: str
    ranker_weights: dict[str, float]
    override_weights: dict[str, float]
    override_threshold: float
    metadata: dict[str, Any]


def build_gate_artifact(
    examples: list[Example],
    *,
    epochs: int = 4,
    gate_epochs: int = 80,
    negative_weight: float = 8.0,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
) -> dict[str, Any]:
    split = split_manual_calibration(examples, manual_calibration_ratio=manual_calibration_ratio, seed=seed)
    ranker_weights = train_perceptron(split.real_train, epochs=epochs, seed=seed)
    train_pool = split.real_train + split.calibration
    train_pred, train_conf = _predict(train_pool, ranker_weights)
    rows = _rows_from_examples(train_pool, train_pred, train_conf)
    override_model = train_override_classifier(rows, epochs=gate_epochs, negative_weight=negative_weight)
    return {
        "schema_version": SCHEMA_VERSION,
        "ranker_model": {
            "type": "stdlib_perceptron_ranker",
            "weights": ranker_weights,
        },
        "override_model": {
            "type": "stdlib_logistic_override_gate",
            "weights": override_model.weights,
            "threshold": override_model.threshold,
        },
        "metadata": {
            "seed": seed,
            "epochs": epochs,
            "gate_epochs": gate_epochs,
            "negative_weight": negative_weight,
            "manual_calibration_ratio": manual_calibration_ratio,
            "override_training_examples": len(rows),
            "train_pool_total": len(train_pool),
        },
    }


def export_gate_artifact(
    examples: list[Example],
    output: str | Path,
    *,
    epochs: int = 4,
    gate_epochs: int = 80,
    negative_weight: float = 8.0,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
) -> dict[str, Any]:
    payload = build_gate_artifact(
        examples,
        epochs=epochs,
        gate_epochs=gate_epochs,
        negative_weight=negative_weight,
        seed=seed,
        manual_calibration_ratio=manual_calibration_ratio,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_gate_artifact(path: str | Path) -> GateArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GateArtifact(
        schema_version=str(payload.get("schema_version")),
        ranker_weights={str(k): float(v) for k, v in ((payload.get("ranker_model") or {}).get("weights") or {}).items()},
        override_weights={str(k): float(v) for k, v in ((payload.get("override_model") or {}).get("weights") or {}).items()},
        override_threshold=float((payload.get("override_model") or {}).get("threshold", 0.5)),
        metadata=dict(payload.get("metadata") or {}),
    )
