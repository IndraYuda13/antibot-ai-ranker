from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.disagreements import mine_disagreements, summarize_disagreements


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


def test_mine_disagreements_labels_negative_and_positive_override_cases():
    examples = [
        _ex("neg", "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]),
        _ex("pos", "manual_label", ["a", "b", "c"], ["a", "c", "b"]),
        _ex("same", "manual_label", ["a", "b", "c"], ["a", "b", "c"]),
    ]
    predictions = {
        "neg": ["a", "c", "b"],
        "pos": ["a", "b", "c"],
        "same": ["a", "b", "c"],
    }
    confidences = {"neg": 0.9, "pos": 0.8, "same": 0.7}

    rows = mine_disagreements(examples, predictions, confidences)

    by_id = {row["case_id"]: row for row in rows}
    assert by_id["neg"]["category"] == "negative_rule_ok_ai_wrong"
    assert by_id["pos"]["category"] == "positive_rule_wrong_ai_ok"
    assert "same" not in by_id


def test_summarize_disagreements_counts_by_category_and_source():
    rows = [
        {"category": "negative_rule_ok_ai_wrong", "source": "accepted_success_raw"},
        {"category": "positive_rule_wrong_ai_ok", "source": "manual_label"},
        {"category": "positive_rule_wrong_ai_ok", "source": "manual_label"},
    ]

    summary = summarize_disagreements(rows)

    assert summary["total"] == 3
    assert summary["by_category"]["positive_rule_wrong_ai_ok"] == 2
    assert summary["by_source"]["manual_label"] == 2
