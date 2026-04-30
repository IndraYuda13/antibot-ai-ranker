from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.override import OverrideExample, build_override_examples, train_override_classifier, predict_override


def _ex(case_id: str, source: str, expected: list[str], rule: list[str]) -> Example:
    return Example(
        case_id=case_id,
        attempt_id=1,
        source=source,
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_build_override_examples_labels_safe_overrides():
    examples = [
        _ex("raw", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]),
        _ex("manual", "manual_label", ["a", "b", "c"], ["a", "c", "b"]),
    ]
    preds = {"raw": ["a", "c", "b"], "manual": ["a", "b", "c"]}
    conf = {"raw": 0.8, "manual": 0.8}
    rows = build_override_examples(examples, preds, conf)

    by_id = {row.case_id: row for row in rows}
    assert by_id["raw"].label == 0
    assert by_id["manual"].label == 1


def test_train_override_classifier_predicts_positive_for_similar_gain_case():
    rows = [
        OverrideExample("bad", {"bias": 1.0, "ai_confidence": 0.8, "source_manual": 0.0, "rule_ai_disagree": 1.0}, 0),
        OverrideExample("good", {"bias": 1.0, "ai_confidence": 0.8, "source_manual": 1.0, "rule_ai_disagree": 1.0}, 1),
    ]
    model = train_override_classifier(rows, epochs=20)
    assert predict_override(model, {"bias": 1.0, "ai_confidence": 0.8, "source_manual": 1.0, "rule_ai_disagree": 1.0}) is True
    assert predict_override(model, {"bias": 1.0, "ai_confidence": 0.8, "source_manual": 0.0, "rule_ai_disagree": 1.0}) is False
