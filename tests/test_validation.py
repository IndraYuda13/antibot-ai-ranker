from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.validation import apply_family_thresholds, validated_family_gate_report


def _ex(case_id: str, question: str, expected: list[str], rule: list[str], source: str = "accepted_success_raw") -> Example:
    return Example(
        case_id=case_id,
        attempt_id=1,
        source=source,
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=[question],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_apply_family_thresholds_uses_family_specific_threshold():
    examples = [
        _ex("num", "one, two, three", ["a", "b", "c"], ["a", "c", "b"]),
        _ex("short", "ice, pan, tea", ["a", "b", "c"], ["a", "b", "c"]),
    ]
    preds = {"num": ["a", "b", "c"], "short": ["a", "c", "b"]}
    conf = {"num": 0.8, "short": 0.2}
    report = apply_family_thresholds(examples, preds, conf, {"number_words": 0.5, "short_words": 0.9})

    assert report["hybrid"]["ok"] == 2


def test_validated_family_gate_report_chooses_thresholds_on_dev_only():
    examples = [_ex(f"r{i}", "ice, pan, tea", ["a", "b", "c"], ["a", "b", "c"]) for i in range(30)]
    report = validated_family_gate_report(examples, epochs=1, seed=7)

    assert report["counts"]["train"] > 0
    assert "selected_thresholds" in report
    assert "test_metrics" in report
