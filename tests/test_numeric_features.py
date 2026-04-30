from __future__ import annotations

from antibot_ai_ranker.features import pair_features, predict_order
from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.train import default_weights


def test_pair_features_exposes_numeric_match():
    values = pair_features("seven", ["7"])
    assert values["numeric_match"] == 1.0


def test_default_ranker_orders_number_words_against_digits(tmp_path):
    ex = Example(
        case_id="num",
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=tmp_path / "x.json",
        question_ocr=["seven, four, six"],
        option_ocr={"a": ["4"], "b": ["6"], "c": ["7"]},
        expected_order=["c", "a", "b"],
    )
    assert predict_order(ex, default_weights()) == ["c", "a", "b"]
