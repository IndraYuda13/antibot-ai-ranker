from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.features import predict_order_scored
from antibot_ai_ranker.train import default_weights


def test_predict_order_scored_returns_order_and_confidence(tmp_path):
    ex = Example(
        case_id="demo",
        attempt_id=1,
        source="manual_label",
        verdict="server_reject_antibot",
        capture_path=tmp_path / "demo.json",
        question_ocr=["tea, pan, ice"],
        option_ocr={"a": ["t3@"], "b": ["p@n"], "c": ["1c3"]},
        expected_order=["a", "b", "c"],
    )
    result = predict_order_scored(ex, default_weights())
    assert result.order == ["a", "b", "c"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.best_score >= result.second_best_score
