from __future__ import annotations

from antibot_ai_ranker.family import classify_tokens, example_family
from antibot_ai_ranker.dataset import Example


def test_classify_tokens_detects_number_words_and_leetspeak():
    assert classify_tokens(["one", "two", "three"]) == "number_words"
    assert classify_tokens(["ice", "pan", "tea"]) == "short_words"
    assert classify_tokens(["0", "2", "3"]) == "numeric"
    assert classify_tokens(["t3@", "p@n", "1c3"]) == "leetspeak"


def test_example_family_uses_first_valid_token_set():
    ex = Example(
        case_id="demo",
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["one, two, three"],
        option_ocr={"a": ["1"], "b": ["2"], "c": ["3"]},
        expected_order=["a", "b", "c"],
    )
    assert example_family(ex) == "number_words"
