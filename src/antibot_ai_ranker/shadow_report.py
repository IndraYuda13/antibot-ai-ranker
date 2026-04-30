from __future__ import annotations

from collections import Counter

from .balanced_validation import split_manual_calibration
from .dataset import Example
from .disagreement_gate import _predict, _rows_from_examples
from .override import train_override_classifier
from .shadow import build_shadow_decision, shadow_decision_to_json
from .train import train_perceptron

REPORT_SCHEMA_VERSION = "antibot-ai-ranker.shadow-report.v1"


def _summary(decisions: list[dict[str, object]]) -> dict[str, object]:
    by_source = Counter(str(item.get("source")) for item in decisions)
    by_family = Counter(str(item.get("family")) for item in decisions)
    would_override = sum(1 for item in decisions if item.get("would_override") is True)
    return {
        "total": len(decisions),
        "would_override": would_override,
        "kept_production_order": len(decisions) - would_override,
        "by_source": dict(sorted(by_source.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def build_shadow_report(
    examples: list[Example],
    *,
    epochs: int = 4,
    gate_epochs: int = 80,
    negative_weight: float = 8.0,
    seed: int = 1337,
    manual_calibration_ratio: float = 0.5,
    limit: int | None = None,
) -> dict[str, object]:
    split = split_manual_calibration(examples, manual_calibration_ratio=manual_calibration_ratio, seed=seed)
    weights = train_perceptron(split.real_train, epochs=epochs, seed=seed)
    train_pool = split.real_train + split.calibration
    train_pred, train_conf = _predict(train_pool, weights)
    rows = _rows_from_examples(train_pool, train_pred, train_conf)
    model = train_override_classifier(rows, epochs=gate_epochs, negative_weight=negative_weight)

    candidates = split.real_test + split.manual_test
    if limit is not None:
        candidates = candidates[:limit]
    pred, conf = _predict(candidates, weights)
    decisions = [
        shadow_decision_to_json(
            build_shadow_decision(
                ex,
                ai_order=pred.get(ex.case_id, []),
                ai_confidence=conf.get(ex.case_id, 0.0),
                override_model=model,
            )
        )
        for ex in candidates
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "shadow_no_submit",
        "no_submit": True,
        "seed": seed,
        "epochs": epochs,
        "gate_epochs": gate_epochs,
        "negative_weight": negative_weight,
        "override_training_examples": len(rows),
        "summary": _summary(decisions),
        "decisions": decisions,
    }
