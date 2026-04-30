from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.disagreement_gate import train_disagreement_gate_report


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


def test_train_disagreement_gate_report_has_training_counts_and_metrics():
    examples = [_ex(f"raw_{i}", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(40)]
    examples += [_ex(f"manual_{i}", "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(20)]

    report = train_disagreement_gate_report(examples, epochs=1, seed=3, gate_epochs=1)

    assert "disagreement_training" in report
    assert report["override_training_examples"] > 0
    assert "real_test_metrics" in report
    assert "manual_test_metrics" in report
