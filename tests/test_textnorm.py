from claimcoin_antibot_ai_ranker.textnorm import clean, similarity


def test_clean_leet_text():
    assert clean("t3@") == "tea"


def test_similarity_exact_is_high():
    assert similarity("tea", "t3@") > 0.9
