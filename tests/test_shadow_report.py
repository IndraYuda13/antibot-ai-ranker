from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.shadow_report import build_shadow_report


def _ex(i: int, source: str, rule: list[str]) -> Example:
    return Example(
        case_id=f"{source}_{i:03d}",
        attempt_id=i,
        source=source,
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=["a", "b", "c"],
        solver_order=rule,
    )


def test_build_shadow_report_contains_schema_and_decisions():
    examples = [_ex(i, "accepted_success_raw", ["a", "b", "c"]) for i in range(30)]
    examples += [_ex(i, "manual_label", ["a", "c", "b"]) for i in range(10)]

    report = build_shadow_report(examples, epochs=1, gate_epochs=1, seed=3)

    assert report["schema_version"] == "antibot-ai-ranker.shadow-report.v1"
    assert report["mode"] == "shadow_no_submit"
    assert report["no_submit"] is True
    assert len(report["decisions"]) > 0
    assert "summary" in report
