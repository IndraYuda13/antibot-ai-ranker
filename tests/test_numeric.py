from __future__ import annotations

from antibot_ai_ranker.numeric import numeric_value, numeric_similarity


def test_numeric_value_handles_digits_words_roman_and_math():
    assert numeric_value("7") == 7
    assert numeric_value("seven") == 7
    assert numeric_value("VII") == 7
    assert numeric_value("3+4") == 7
    assert numeric_value("8-1") == 7
    assert numeric_value("2*4") == 8


def test_numeric_value_handles_leet_number_words():
    assert numeric_value("s3v3n") == 7
    assert numeric_value("thr33") == 3
    assert numeric_value("f0vr") == 4


def test_numeric_similarity_matches_equivalent_forms():
    assert numeric_similarity("seven", "7") == 1.0
    assert numeric_similarity("4", "four") == 1.0
    assert numeric_similarity("six", "5") == 0.0
