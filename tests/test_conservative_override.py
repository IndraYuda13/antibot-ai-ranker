from __future__ import annotations

from dataclasses import replace

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.override import OverrideModel, calibrate_override_threshold
from antibot_ai_ranker.override_validation import conservative_override_gate_report


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


def test_calibrate_override_threshold_prefers_accepted_safe_cutoff():
    examples = [
        _ex("raw", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]),
        _ex("manual", "manual_label", ["a", "b", "c"], ["a", "c", "b"]),
    ]
    predictions = {"raw": ["a", "c", "b"], "manual": ["a", "b", "c"]}
    confidences = {"raw": 0.8, "manual": 0.8}
    # source_manual weight makes manual probability high, but raw still above 0.5.
    model = OverrideModel({"bias": 0.1, "source_manual": 2.0})

    calibrated = calibrate_override_threshold(model, examples, predictions, confidences, thresholds=[0.5, 0.8, 0.95])

    assert calibrated.threshold == 0.8


def test_conservative_override_gate_report_includes_selected_threshold():
    examples = [_ex(f"raw_{i}", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(40)]
    examples += [_ex(f"manual_{i}", "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(20)]

    report = conservative_override_gate_report(examples, epochs=1, seed=7, override_epochs=1)

    assert "selected_override_threshold" in report
    assert 0.0 <= report["selected_override_threshold"] <= 1.0
    assert "manual_test_metrics" in report
