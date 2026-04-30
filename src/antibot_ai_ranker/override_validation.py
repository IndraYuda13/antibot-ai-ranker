from __future__ import annotations

from .balanced_validation import split_manual_calibration
from .benchmark import benchmark_orders
from .dataset import Example
from .features import predict_order_scored
from .override import build_override_examples, override_features, predict_override, train_override_classifier
from .train import train_perceptron


def _predict(examples: list[Example], weights: dict[str, float]) -> tuple[dict[str, list[str]], dict[str, float]]:
    predictions: dict[str, list[str]] = {}
    confidences: dict[str, float] = {}
    for ex in examples:
        pred = predict_order_scored(ex, weights)
        predictions[ex.case_id] = pred.order
        confidences[ex.case_id] = pred.confidence
    return predictions, confidences


def _apply_override_model(examples: list[Example], predictions: dict[str, list[str]], confidences: dict[str, float], model) -> dict[str, list[str]]:
    gated: dict[str, list[str]] = {}
    for ex in examples:
        ai_order = list(predictions.get(ex.case_id) or [])
        features = override_features(ex, ai_order, float(confidences.get(ex.case_id, 0.0)))
        gated[ex.case_id] = ai_order if predict_override(model, features) else list(ex.solver_order or [])
    return gated


def override_gate_report(
    examples: list[Example],
    *,
    epochs: int = 8,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
    override_epochs: int = 25,
) -> dict[str, object]:
    split = split_manual_calibration(examples, manual_calibration_ratio=manual_calibration_ratio, seed=seed)
    ranker_weights = train_perceptron(split.train, epochs=epochs, seed=seed)

    cal_pred, cal_conf = _predict(split.calibration, ranker_weights)
    override_rows = build_override_examples(split.calibration, cal_pred, cal_conf)
    override_model = train_override_classifier(override_rows, epochs=override_epochs)

    real_pred, real_conf = _predict(split.real_test, ranker_weights)
    manual_pred, manual_conf = _predict(split.manual_test, ranker_weights)
    cal_gated = _apply_override_model(split.calibration, cal_pred, cal_conf, override_model)
    real_gated = _apply_override_model(split.real_test, real_pred, real_conf, override_model)
    manual_gated = _apply_override_model(split.manual_test, manual_pred, manual_conf, override_model)

    # Treat gated predictions as AI predictions with confidence 1.0 so benchmark's
    # hybrid column equals the override model's final order.
    return {
        "seed": seed,
        "epochs": epochs,
        "manual_calibration_ratio": manual_calibration_ratio,
        "override_training_examples": len(override_rows),
        "counts": {
            "real_train": len(split.real_train),
            "real_dev": len(split.real_dev),
            "real_test": len(split.real_test),
            "manual_calibration": len(split.manual_calibration),
            "manual_test": len(split.manual_test),
            "calibration_total": len(split.calibration),
            "total": len(split.real_train) + len(split.real_dev) + len(split.real_test) + len(split.manual_calibration) + len(split.manual_test),
        },
        "calibration_metrics": benchmark_orders(split.calibration, cal_gated, ai_confidences={k: 1.0 for k in cal_gated}, threshold=0.5),
        "real_test_metrics": benchmark_orders(split.real_test, real_gated, ai_confidences={k: 1.0 for k in real_gated}, threshold=0.5),
        "manual_test_metrics": benchmark_orders(split.manual_test, manual_gated, ai_confidences={k: 1.0 for k in manual_gated}, threshold=0.5),
    }
