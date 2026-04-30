from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .dataset import Example
from .family import example_family


@dataclass(frozen=True)
class OverrideExample:
    case_id: str
    features: dict[str, float]
    label: int


@dataclass(frozen=True)
class OverrideModel:
    weights: dict[str, float]
    threshold: float = 0.5


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def override_features(example: Example, ai_order: list[str], ai_confidence: float) -> dict[str, float]:
    rule_order = list(example.solver_order or [])
    family = example_family(example)
    return {
        "bias": 1.0,
        "ai_confidence": float(ai_confidence),
        "rule_ai_disagree": 1.0 if rule_order and ai_order and rule_order != ai_order else 0.0,
        "source_manual": 1.0 if example.source == "manual_label" else 0.0,
        "source_accepted_raw": 1.0 if example.source == "accepted_success_raw" else 0.0,
        "family_words": 1.0 if family == "words" else 0.0,
        "family_short_words": 1.0 if family == "short_words" else 0.0,
        "family_leetspeak": 1.0 if family == "leetspeak" else 0.0,
        "family_numeric": 1.0 if family == "numeric" else 0.0,
        "family_number_words": 1.0 if family == "number_words" else 0.0,
    }


def build_override_examples(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
) -> list[OverrideExample]:
    rows: list[OverrideExample] = []
    for ex in examples:
        rule_order = list(ex.solver_order or [])
        ai_order = list(ai_predictions.get(ex.case_id) or [])
        expected = list(ex.expected_order)
        if not rule_order or not ai_order or rule_order == ai_order:
            continue
        rule_ok = rule_order == expected
        ai_ok = ai_order == expected
        if rule_ok and not ai_ok:
            label = 0
        elif not rule_ok and ai_ok:
            label = 1
        else:
            # No useful supervised signal for override decision.
            continue
        rows.append(OverrideExample(ex.case_id, override_features(ex, ai_order, float(confidences.get(ex.case_id, 0.0))), label))
    return rows


def _score(weights: Mapping[str, float], features: Mapping[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def train_override_classifier(rows: list[OverrideExample], *, epochs: int = 20, lr: float = 0.4, negative_weight: float = 3.0) -> OverrideModel:
    weights: dict[str, float] = {}
    for _ in range(max(1, epochs)):
        for row in rows:
            pred = _sigmoid(_score(weights, row.features))
            weight = negative_weight if row.label == 0 else 1.0
            error = (row.label - pred) * weight
            for name, value in row.features.items():
                weights[name] = weights.get(name, 0.0) + lr * error * value
    return OverrideModel(weights)


def override_probability(model: OverrideModel, features: Mapping[str, float]) -> float:
    return _sigmoid(_score(model.weights, features))


def predict_override(model: OverrideModel, features: Mapping[str, float]) -> bool:
    return override_probability(model, features) >= model.threshold


def calibrate_override_threshold(
    model: OverrideModel,
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
    *,
    thresholds: list[float] | None = None,
    min_accepted_accuracy: float = 100.0,
) -> OverrideModel:
    thresholds = thresholds or [i / 100 for i in range(0, 101)]
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        accepted_total = accepted_ok = manual_total = manual_ok = 0
        candidate = OverrideModel(dict(model.weights), threshold=float(threshold))
        for ex in examples:
            ai_order = list(ai_predictions.get(ex.case_id) or [])
            final_order = ai_order if predict_override(candidate, override_features(ex, ai_order, float(confidences.get(ex.case_id, 0.0)))) else list(ex.solver_order or [])
            ok = final_order == list(ex.expected_order)
            if ex.source == "accepted_success_raw":
                accepted_total += 1
                accepted_ok += int(ok)
            elif ex.source == "manual_label":
                manual_total += 1
                manual_ok += int(ok)
        accepted_accuracy = 100.0 if not accepted_total else (accepted_ok / accepted_total) * 100
        manual_accuracy = 0.0 if not manual_total else (manual_ok / manual_total) * 100
        safe = accepted_accuracy >= min_accepted_accuracy
        score = (1.0 if safe else 0.0, manual_accuracy, accepted_accuracy)
        if best is None or score > best:
            best = score
            best_threshold = float(threshold)
    return OverrideModel(dict(model.weights), threshold=best_threshold)
