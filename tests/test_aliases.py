from __future__ import annotations

from antibot_ai_ranker.aliases import alias_similarity, canonical_alias


def test_canonical_alias_handles_short_word_ocr_confusions():
    assert canonical_alias("2p") == "zip"
    assert canonical_alias("200") == "zoo"
    assert canonical_alias("20r") == "zor"
    assert canonical_alias("mc") == "arc"
    assert canonical_alias("cir") == "or"


def test_canonical_alias_handles_leetspeak_live_confusions():
    assert canonical_alias("teg") == "424"
    assert canonical_alias("t03") == "te"
    assert canonical_alias("mal") == "mel"
    assert canonical_alias("dnt") == "lem"


def test_alias_similarity_matches_known_equivalent_forms():
    assert alias_similarity("zip", "2p") == 1.0
    assert alias_similarity("424", "teg") == 1.0
    assert alias_similarity("lem", "dnt") == 1.0
    assert alias_similarity("zip", "zoo") == 0.0
