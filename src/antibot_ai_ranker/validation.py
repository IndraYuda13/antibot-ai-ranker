from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from .benchmark import benchmark_orders, choose_hybrid_order
from .confidence import sweep_family_thresholds
from .dataset import Example
from .family import example_family
from .features import predict_order_scored
from .splits import split_examples
from .train import train_perceptron


def apply_family_thresholds(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
    thresholds: Mapping[str, float],
    *,
    default_threshold: float = 1.0,
) -> dict[str, object]:
    adjusted_predictions: dict[str, list[str]] = {}
    adjusted_confidences: dict[str, float] = {}
    for ex in examples:
        family = example_family(ex)
        threshold = float(thresholds.get(family, default_threshold))
        adjusted_predictions[ex.case_id] = list(ai_predictions.get(ex.case_id) or [])
        adjusted_confidences[ex.case_id] = float(confidences.get(ex.case_id, 0.0))
        # benchmark_orders accepts one global threshold. Force per-example decision
        # by rewriting confidence to either pass or fail against threshold 0.5.
        if adjusted_confidences[ex.case_id] >= threshold:
            adjusted_confidences[ex.case_id] = 1.0
        else:
            adjusted_confidences[ex.case_id] = 0.0
    return benchmark_orders(examples, adjusted_predictions, ai_confidences=adjusted_confidences, threshold=0.5)


def _predict_all(examples: list[Example], weights: dict[str, float]) -> tuple[dict[str, list[str]], dict[str, float]]:
    predictions: dict[str, list[str]] = {}
    confidences: dict[str, float] = {}
    for ex in examples:
        pred = predict_order_scored(ex, weights)
        predictions[ex.case_id] = pred.order
        confidences[ex.case_id] = pred.confidence
    return predictions, confidences


def validated_family_gate_report(
    examples: list[Example],
    *,
    epochs: int = 8,
    seed: int = 1337,
    holdout_sources: set[str] | None = None,
) -> dict[str, object]:
    split = split_examples(examples, seed=seed, holdout_sources=holdout_sources or set())
    weights = train_perceptron(split.train, epochs=epochs, seed=seed)

    dev_predictions, dev_confidences = _predict_all(split.dev, weights)
    dev_sweep = sweep_family_thresholds(split.dev, dev_predictions, dev_confidences)
    thresholds = dev_sweep.get("best_thresholds", {})

    train_predictions, train_confidences = _predict_all(split.train, weights)
    test_predictions, test_confidences = _predict_all(split.test, weights)
    held_predictions, held_confidences = _predict_all(split.heldout, weights)

    return {
        "seed": seed,
        "epochs": epochs,
        "counts": {
            "train": len(split.train),
            "dev": len(split.dev),
            "test": len(split.test),
            "heldout": len(split.heldout),
            "total": len(split.all_examples),
        },
        "selected_thresholds": thresholds,
        "dev_selection": dev_sweep,
        "train_metrics": apply_family_thresholds(split.train, train_predictions, train_confidences, thresholds),
        "test_metrics": apply_family_thresholds(split.test, test_predictions, test_confidences, thresholds),
        "heldout_metrics": apply_family_thresholds(split.heldout, held_predictions, held_confidences, thresholds) if split.heldout else {"rule": {"total": 0, "ok": 0, "wrong": 0, "accuracy": 0.0}, "ai": {"total": 0, "ok": 0, "wrong": 0, "accuracy": 0.0}, "hybrid": {"total": 0, "ok": 0, "wrong": 0, "accuracy": 0.0}},
    }
