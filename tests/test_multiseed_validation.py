from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.multiseed_validation import summarize_multiseed, multiseed_override_report


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


def test_summarize_multiseed_reports_min_mean_max():
    summary = summarize_multiseed([
        {"manual_test_metrics": {"hybrid": {"accuracy": 50.0}}, "real_test_metrics": {"hybrid": {"accuracy": 100.0}}},
        {"manual_test_metrics": {"hybrid": {"accuracy": 75.0}}, "real_test_metrics": {"hybrid": {"accuracy": 90.0}}},
    ])
    assert summary["manual_hybrid_accuracy"] == {"min": 50.0, "mean": 62.5, "max": 75.0}
    assert summary["real_hybrid_accuracy"] == {"min": 90.0, "mean": 95.0, "max": 100.0}


def test_multiseed_override_report_runs_requested_seeds():
    examples = [_ex(i, "accepted_success_raw") for i in range(40)] + [_ex(i, "manual_label") for i in range(20)]
    report = multiseed_override_report(examples, seeds=[1, 2], epochs=1, override_epochs=1)
    assert report["seeds"] == [1, 2]
    assert len(report["runs"]) == 2
    assert "summary" in report
