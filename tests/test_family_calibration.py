from __future__ import annotations

from antibot_ai_ranker.confidence import sweep_family_thresholds
from antibot_ai_ranker.dataset import Example


def _ex(case_id: str, question: str, expected: list[str], rule: list[str]) -> Example:
    return Example(
        case_id=case_id,
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=[question],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_sweep_family_thresholds_returns_thresholds_by_family():
    examples = [
        _ex("num", "one, two, three", ["a", "b", "c"], ["a", "c", "b"]),
        _ex("short", "ice, pan, tea", ["a", "b", "c"], ["a", "b", "c"]),
    ]
    preds = {"num": ["a", "b", "c"], "short": ["a", "c", "b"]}
    conf = {"num": 0.8, "short": 0.2}
    report = sweep_family_thresholds(examples, preds, conf, thresholds=[0.0, 0.5, 0.9])

    assert "number_words" in report["families"]
    assert "short_words" in report["families"]
    assert report["best_thresholds"]["number_words"] == 0.5
    assert report["best_thresholds"]["short_words"] == 0.9
