from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.override_validation import override_gate_report


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


def test_override_gate_report_has_manual_test_metrics():
    examples = [_ex(i, "accepted_success_raw") for i in range(40)] + [_ex(i, "manual_label") for i in range(20)]
    report = override_gate_report(examples, epochs=1, seed=5, manual_calibration_ratio=0.5)

    assert report["counts"]["manual_test"] == 10
    assert "manual_test_metrics" in report
    assert "override_training_examples" in report
