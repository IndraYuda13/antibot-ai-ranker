from __future__ import annotations

from antibot_ai_ranker.balanced_validation import balanced_manual_gate_report, split_manual_calibration
from antibot_ai_ranker.dataset import Example


def _ex(i: int, source: str) -> Example:
    return Example(
        case_id=f"{source}_{i:03d}",
        attempt_id=i,
        source=source,
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=["a", "b", "c"],
        solver_order=["a", "c", "b"] if source == "manual_label" else ["a", "b", "c"],
    )


def test_split_manual_calibration_keeps_separate_manual_test():
    examples = [_ex(i, "accepted_success_raw") for i in range(20)] + [_ex(i, "manual_label") for i in range(10)]
    split = split_manual_calibration(examples, manual_calibration_ratio=0.4, seed=7)

    assert len(split.manual_calibration) == 4
    assert len(split.manual_test) == 6
    assert {ex.case_id for ex in split.manual_calibration}.isdisjoint({ex.case_id for ex in split.manual_test})


def test_balanced_manual_gate_report_reports_manual_calibration_and_test():
    examples = [_ex(i, "accepted_success_raw") for i in range(40)] + [_ex(i, "manual_label") for i in range(20)]
    report = balanced_manual_gate_report(examples, epochs=1, seed=3, manual_calibration_ratio=0.5)

    assert report["counts"]["manual_calibration"] == 10
    assert report["counts"]["manual_test"] == 10
    assert "manual_test_metrics" in report
    assert "selected_thresholds" in report
