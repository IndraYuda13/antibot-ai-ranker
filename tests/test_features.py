from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.features import predict_order
from antibot_ai_ranker.train import default_weights


def test_predict_order_simple_three_options(tmp_path):
    ex = Example(
        case_id="demo",
        attempt_id=1,
        source="manual_label",
        verdict="server_reject_antibot",
        capture_path=tmp_path / "demo.json",
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=["a", "b", "c"],
    )
    assert predict_order(ex, default_weights()) == ["a", "b", "c"]
