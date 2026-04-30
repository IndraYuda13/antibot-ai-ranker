from __future__ import annotations

from .dataset import Example
from .features import token_sets
from .textnorm import clean

NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}
ANIMAL_WORDS = {"ant", "camel", "cat", "cow", "crab", "deer", "dog", "duck", "elephant", "fox", "lion", "monkey", "mouse", "panda", "rabbit", "tiger"}


def classify_tokens(tokens: list[str]) -> str:
    cleaned = [clean(token) for token in tokens if clean(token)]
    raw = [str(token).lower() for token in tokens]
    if not cleaned:
        return "unknown"
    if all(str(token).strip().isdigit() for token in tokens):
        return "numeric"
    if all(token in NUMBER_WORDS for token in cleaned):
        return "number_words"
    if any(any(ch in token for ch in "@01345789") for token in raw):
        return "leetspeak"
    if all(token in ANIMAL_WORDS for token in cleaned):
        return "animals"
    if max(len(token) for token in cleaned) <= 4:
        return "short_words"
    return "words"


def example_family(example: Example) -> str:
    sets = token_sets(example.question_ocr, len(example.option_ocr))
    if not sets:
        return "unknown"
    return classify_tokens(sets[0])
