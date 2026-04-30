from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.features import pair_features, predict_order
from antibot_ai_ranker.train import default_weights


def test_pair_features_exposes_alias_match():
    values = pair_features("zip", ["2p"])
    assert values["alias_match"] == 1.0


def test_default_ranker_orders_short_alias_family(tmp_path):
    ex = Example(
        case_id="zip",
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=tmp_path / "x.json",
        question_ocr=["zul, zip, zig"],
        option_ocr={"a": ["ug"], "b": ["zi"], "c": ["2p"]},
        expected_order=["b", "c", "a"],
    )
    assert predict_order(ex, default_weights()) == ["b", "c", "a"]


def test_default_ranker_orders_leetspeak_alias_family(tmp_path):
    ex = Example(
        case_id="leet",
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=tmp_path / "x.json",
        question_ocr=["424, try, te"],
        option_ocr={"a": ["t03"], "b": ["teg"], "c": ["at"]},
        expected_order=["b", "c", "a"],
    )
    assert predict_order(ex, default_weights()) == ["b", "c", "a"]
