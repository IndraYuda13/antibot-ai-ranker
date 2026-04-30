from __future__ import annotations

from antibot_ai_ranker.confidence import order_confidence, sweep_thresholds
from antibot_ai_ranker.dataset import Example


def _ex(case_id: str, expected: list[str], rule: list[str]) -> Example:
    return Example(
        case_id=case_id,
        attempt_id=1,
        source="manual_label",
        verdict="server_reject_antibot",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["one, two, three"],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_order_confidence_uses_score_gap():
    assert order_confidence(best_score=10.0, second_best_score=9.0) < order_confidence(best_score=10.0, second_best_score=2.0)
    assert order_confidence(best_score=0.0, second_best_score=0.0) == 0.0


def test_sweep_thresholds_reports_best_threshold():
    examples = [
        _ex("a", ["a", "b", "c"], ["a", "b", "c"]),
        _ex("b", ["a", "b", "c"], ["a", "c", "b"]),
    ]
    ai_predictions = {"a": ["a", "c", "b"], "b": ["a", "b", "c"]}
    confidences = {"a": 0.2, "b": 0.9}
    report = sweep_thresholds(examples, ai_predictions, confidences, thresholds=[0.0, 0.5, 0.95])

    assert report["best"]["threshold"] == 0.5
    assert report["best"]["hybrid"]["ok"] == 2
    assert len(report["thresholds"]) == 3
