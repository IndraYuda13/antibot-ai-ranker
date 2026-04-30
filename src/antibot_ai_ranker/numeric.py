from __future__ import annotations

import re

WORD_TO_NUMBER = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
LEET_WORD_FIXES = {
    "z3ro": "zero",
    "thr33": "three",
    "s3v3n": "seven",
    "3ight": "eight",
    "fiv3": "five",
    "f0vr": "four",
    "fovr": "four",
    "f0ur": "four",
    "fovr": "four",
    "slx": "six",
    "s1x": "six",
    "t3n": "ten",
}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "@": "a"})


def _clean(value: str) -> str:
    return re.sub(r"[^a-z0-9+*x\-]+", "", str(value).strip().lower())


def _eval_expr(value: str) -> int | None:
    m = re.fullmatch(r"(\d+)([+\-*x])(\d+)", value)
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def numeric_value(value: str) -> int | None:
    raw = _clean(value)
    if not raw:
        return None
    expr = _eval_expr(raw)
    if expr is not None:
        return expr
    if raw.isdigit():
        return int(raw)
    if raw in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[raw]
    if raw in ROMAN:
        return ROMAN[raw]
    fixed = LEET_WORD_FIXES.get(raw) or LEET_WORD_FIXES.get(raw.translate(LEET))
    if fixed and fixed in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[fixed]
    translated = raw.translate(LEET)
    if translated in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[translated]
    if translated in ROMAN:
        return ROMAN[translated]
    return None


def numeric_similarity(a: str, b: str) -> float:
    av = numeric_value(a)
    bv = numeric_value(b)
    if av is None or bv is None:
        return 0.0
    return 1.0 if av == bv else 0.0
