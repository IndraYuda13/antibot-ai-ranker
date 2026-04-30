from __future__ import annotations

import re

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s", "+": "t", "|": "l"})


def clean(text: str) -> str:
    value = str(text or "").strip().lower().translate(LEET)
    return re.sub(r"[^a-z0-9]+", "", value)


def char_ngrams(text: str, n: int = 2) -> set[str]:
    value = clean(text)
    if len(value) <= n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: str, b: str) -> float:
    ca, cb = clean(a), clean(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return 0.5 * jaccard(char_ngrams(ca, 2), char_ngrams(cb, 2)) + 0.5 * jaccard(set(ca), set(cb))
