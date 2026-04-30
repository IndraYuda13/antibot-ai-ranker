from __future__ import annotations

import random
from dataclasses import dataclass

from .confidence import sweep_family_thresholds
from .dataset import Example
from .features import predict_order_scored
from .splits import split_examples
from .train import train_perceptron
from .validation import apply_family_thresholds


@dataclass(frozen=True)
class BalancedManualSplit:
    real_train: list[Example]
    real_dev: list[Example]
    real_test: list[Example]
    manual_calibration: list[Example]
    manual_test: list[Example]

    @property
    def train(self) -> list[Example]:
        return self.real_train

    @property
    def calibration(self) -> list[Example]:
        return self.real_dev + self.manual_calibration


def split_manual_calibration(
    examples: list[Example],
    *,
    manual_calibration_ratio: float = 0.5,
    seed: int = 1337,
) -> BalancedManualSplit:
    manual = sorted([ex for ex in examples if ex.source == "manual_label"], key=lambda ex: ex.case_id)
    non_manual = [ex for ex in examples if ex.source != "manual_label"]
    rng = random.Random(seed)
    manual_shuffled = manual[:]
    rng.shuffle(manual_shuffled)
    cut = int(len(manual_shuffled) * manual_calibration_ratio)
    base = split_examples(non_manual, seed=seed)
    return BalancedManualSplit(
        real_train=base.train,
        real_dev=base.dev,
        real_test=base.test,
        manual_calibration=manual_shuffled[:cut],
        manual_test=manual_shuffled[cut:],
    )


def _predict(examples: list[Example], weights: dict[str, float]) -> tuple[dict[str, list[str]], dict[str, float]]:
    predictions: dict[str, list[str]] = {}
    confidences: dict[str, float] = {}
    for ex in examples:
        pred = predict_order_scored(ex, weights)
        predictions[ex.case_id] = pred.order
        confidences[ex.case_id] = pred.confidence
    return predictions, confidences


def balanced_manual_gate_report(
    examples: list[Example],
    *,
    epochs: int = 8,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
) -> dict[str, object]:
    split = split_manual_calibration(examples, manual_calibration_ratio=manual_calibration_ratio, seed=seed)
    weights = train_perceptron(split.train, epochs=epochs, seed=seed)

    cal_pred, cal_conf = _predict(split.calibration, weights)
    sweep = sweep_family_thresholds(split.calibration, cal_pred, cal_conf)
    thresholds = sweep.get("best_thresholds", {})

    real_test_pred, real_test_conf = _predict(split.real_test, weights)
    manual_test_pred, manual_test_conf = _predict(split.manual_test, weights)
    cal_test_pred, cal_test_conf = _predict(split.calibration, weights)

    return {
        "seed": seed,
        "epochs": epochs,
        "manual_calibration_ratio": manual_calibration_ratio,
        "counts": {
            "real_train": len(split.real_train),
            "real_dev": len(split.real_dev),
            "real_test": len(split.real_test),
            "manual_calibration": len(split.manual_calibration),
            "manual_test": len(split.manual_test),
            "calibration_total": len(split.calibration),
            "total": len(split.real_train) + len(split.real_dev) + len(split.real_test) + len(split.manual_calibration) + len(split.manual_test),
        },
        "selected_thresholds": thresholds,
        "calibration_metrics": apply_family_thresholds(split.calibration, cal_test_pred, cal_test_conf, thresholds),
        "real_test_metrics": apply_family_thresholds(split.real_test, real_test_pred, real_test_conf, thresholds),
        "manual_test_metrics": apply_family_thresholds(split.manual_test, manual_test_pred, manual_test_conf, thresholds),
        "dev_selection": sweep,
    }
