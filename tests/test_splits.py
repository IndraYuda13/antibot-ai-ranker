from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.splits import split_examples, train_dev_test_report


def _ex(i: int, source: str = "accepted_success_raw") -> Example:
    return Example(
        case_id=f"case_{i:03d}",
        attempt_id=i,
        source=source,
        verdict="accepted_success",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["one, two, three"],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=["a", "b", "c"],
    )


def test_split_examples_is_deterministic_and_complete():
    examples = [_ex(i) for i in range(30)]
    split = split_examples(examples, seed=42, train_ratio=0.6, dev_ratio=0.2)

    assert len(split.train) == 18
    assert len(split.dev) == 6
    assert len(split.test) == 6
    assert sorted(x.case_id for x in split.all_examples) == [x.case_id for x in examples]
    assert split_examples(examples, seed=42).test[0].case_id == split_examples(examples, seed=42).test[0].case_id


def test_train_dev_test_report_keeps_manual_labels_out_of_train_by_default():
    examples = [_ex(i) for i in range(10)] + [_ex(100 + i, source="manual_label") for i in range(5)]
    report = train_dev_test_report(examples, seed=1, holdout_sources={"manual_label"})

    assert report["counts"]["heldout"] == 5
    assert report["counts"]["train"] > 0
    assert "manual_label" in report["heldout_metrics"]["by_source"]
