from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.fast_disagreement_multiseed import fast_disagreement_multiseed_report, summarize_runs


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


def test_summarize_runs_returns_min_mean_max():
    summary = summarize_runs([
        {"real_hybrid_accuracy": 100.0, "manual_hybrid_accuracy": 80.0},
        {"real_hybrid_accuracy": 98.0, "manual_hybrid_accuracy": 70.0},
    ])
    assert summary["real_hybrid_accuracy"] == {"min": 98.0, "mean": 99.0, "max": 100.0}
    assert summary["manual_hybrid_accuracy"] == {"min": 70.0, "mean": 75.0, "max": 80.0}


def test_fast_disagreement_multiseed_report_runs_requested_seeds():
    examples = [_ex(f"raw_{i}", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(30)]
    examples += [_ex(f"manual_{i}", "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(12)]
    report = fast_disagreement_multiseed_report(examples, seeds=[1, 2], epochs=1, gate_epochs=1)
    assert report["seeds"] == [1, 2]
    assert len(report["runs"]) == 2
    assert "summary" in report
