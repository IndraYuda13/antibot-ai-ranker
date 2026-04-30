from __future__ import annotations

from antibot_ai_ranker.benchmark import benchmark_orders, choose_hybrid_order
from antibot_ai_ranker.dataset import Example


def _ex(expected: list[str], rule: list[str], ai: list[str]) -> Example:
    return Example(
        case_id="case",
        attempt_id=1,
        source="manual_label",
        verdict="server_reject_antibot",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["one, two, three"],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_choose_hybrid_prefers_ai_only_when_confident_and_different():
    assert choose_hybrid_order(["a", "b"], ["b", "a"], ai_confidence=0.95, threshold=0.9) == ["b", "a"]
    assert choose_hybrid_order(["a", "b"], ["b", "a"], ai_confidence=0.50, threshold=0.9) == ["a", "b"]


def test_benchmark_orders_reports_rule_ai_and_hybrid_accuracy():
    examples = [
        _ex(["a", "b", "c"], ["a", "b", "c"], ["a", "c", "b"]),
        _ex(["a", "b", "c"], ["a", "c", "b"], ["a", "b", "c"]),
    ]
    ai_predictions = {"case": ["a", "b", "c"]}
    report = benchmark_orders(examples, ai_predictions, ai_confidences={"case": 1.0}, threshold=0.9)

    assert report["rule"]["total"] == 2
    assert report["ai"]["total"] == 2
    assert report["hybrid"]["total"] == 2
    assert "manual_label" in report["by_source"]
