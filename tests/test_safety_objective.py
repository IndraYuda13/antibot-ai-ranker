from __future__ import annotations

from antibot_ai_ranker.balanced_validation import safety_score, select_safety_thresholds
from antibot_ai_ranker.dataset import Example


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


def test_safety_score_penalizes_accepted_raw_regression_more_than_manual_gain():
    safe = safety_score(accepted_regressions=0, manual_gains=2, manual_regressions=0, accepted_penalty=10.0)
    unsafe = safety_score(accepted_regressions=1, manual_gains=2, manual_regressions=0, accepted_penalty=10.0)
    assert safe > unsafe


def test_select_safety_thresholds_prefers_no_accepted_regression():
    examples = [
        _ex("raw", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]),
        _ex("manual", "manual_label", ["a", "b", "c"], ["a", "c", "b"]),
    ]
    preds = {"raw": ["a", "c", "b"], "manual": ["a", "b", "c"]}
    conf = {"raw": 0.4, "manual": 0.9}
    result = select_safety_thresholds(examples, preds, conf, thresholds=[0.0, 0.5])
    assert result["thresholds"]["short_words"] == 0.5
    assert result["metrics"]["hybrid"]["ok"] == 2
