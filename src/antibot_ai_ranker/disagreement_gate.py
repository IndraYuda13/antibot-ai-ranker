from __future__ import annotations

from .balanced_validation import split_manual_calibration
from .benchmark import benchmark_orders
from .dataset import Example
from .disagreements import mine_disagreements, summarize_disagreements
from .features import predict_order_scored
from .override import OverrideExample, OverrideModel, override_features, predict_override, train_override_classifier
from .train import train_perceptron


def _predict(examples: list[Example], weights: dict[str, float]) -> tuple[dict[str, list[str]], dict[str, float]]:
    predictions: dict[str, list[str]] = {}
    confidences: dict[str, float] = {}
    for ex in examples:
        pred = predict_order_scored(ex, weights)
        predictions[ex.case_id] = pred.order
        confidences[ex.case_id] = pred.confidence
    return predictions, confidences


def _rows_from_examples(examples: list[Example], predictions: dict[str, list[str]], confidences: dict[str, float]) -> list[OverrideExample]:
    rows: list[OverrideExample] = []
    for ex in examples:
        rule_order = list(ex.solver_order or [])
        ai_order = list(predictions.get(ex.case_id) or [])
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
            continue
        rows.append(OverrideExample(ex.case_id, override_features(ex, ai_order, float(confidences.get(ex.case_id, 0.0))), label))
    return rows


def _apply(examples: list[Example], predictions: dict[str, list[str]], confidences: dict[str, float], model: OverrideModel) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ex in examples:
        ai_order = list(predictions.get(ex.case_id) or [])
        feats = override_features(ex, ai_order, float(confidences.get(ex.case_id, 0.0)))
        out[ex.case_id] = ai_order if predict_override(model, feats) else list(ex.solver_order or [])
    return out


def train_disagreement_gate_report(
    examples: list[Example],
    *,
    epochs: int = 8,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
    gate_epochs: int = 50,
    negative_weight: float = 8.0,
) -> dict[str, object]:
    split = split_manual_calibration(examples, manual_calibration_ratio=manual_calibration_ratio, seed=seed)
    weights = train_perceptron(split.train, epochs=epochs, seed=seed)
    train_pool = split.real_train + split.calibration
    train_pred, train_conf = _predict(train_pool, weights)
    rows = _rows_from_examples(train_pool, train_pred, train_conf)
    mined = mine_disagreements(train_pool, train_pred, train_conf)
    model = train_override_classifier(rows, epochs=gate_epochs, negative_weight=negative_weight)

    real_pred, real_conf = _predict(split.real_test, weights)
    manual_pred, manual_conf = _predict(split.manual_test, weights)
    cal_pred, cal_conf = _predict(split.calibration, weights)
    real_gated = _apply(split.real_test, real_pred, real_conf, model)
    manual_gated = _apply(split.manual_test, manual_pred, manual_conf, model)
    cal_gated = _apply(split.calibration, cal_pred, cal_conf, model)

    return {
        "seed": seed,
        "epochs": epochs,
        "manual_calibration_ratio": manual_calibration_ratio,
        "gate_epochs": gate_epochs,
        "negative_weight": negative_weight,
        "disagreement_training": summarize_disagreements(mined),
        "override_training_examples": len(rows),
        "counts": {
            "real_train": len(split.real_train),
            "real_dev": len(split.real_dev),
            "real_test": len(split.real_test),
            "manual_calibration": len(split.manual_calibration),
            "manual_test": len(split.manual_test),
            "calibration_total": len(split.calibration),
            "train_pool_total": len(train_pool),
            "total": len(split.real_train) + len(split.real_dev) + len(split.real_test) + len(split.manual_calibration) + len(split.manual_test),
        },
        "calibration_metrics": benchmark_orders(split.calibration, cal_gated, ai_confidences={k: 1.0 for k in cal_gated}, threshold=0.5),
        "real_test_metrics": benchmark_orders(split.real_test, real_gated, ai_confidences={k: 1.0 for k in real_gated}, threshold=0.5),
        "manual_test_metrics": benchmark_orders(split.manual_test, manual_gated, ai_confidences={k: 1.0 for k in manual_gated}, threshold=0.5),
    }
