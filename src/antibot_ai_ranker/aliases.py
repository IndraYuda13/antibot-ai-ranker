from __future__ import annotations

from .textnorm import clean

ALIASES = {
    # Short word / zoo-zip-zig family.
    "2p": "zip",
    "200": "zoo",
    "20r": "zor",
    "zi": "zul",
    "ug": "zig",
    # Pan/pay/ple family.
    "pen": "pan",
    "pey": "pay",
    "poy": "pay",
    "pis": "ple",
    # Arc/cir/row shape family.
    "mc": "arc",
    "cir": "or",
    "dr": "or",
    # Leetspeak/live OCR families.
    "424": "424",
    "te": "te",
    "pom": "pom",
    "mel": "mel",
    "lem": "lem",
    "teg": "424",
    "189": "424",
    "129": "424",
    "t03": "te",
    "+03": "te",
    "tu3": "te",
    "at": "try",
    "ih": "try",
    "we": "try",
    "mal": "mel",
    "dnt": "lem",
    "bn": "lem",
    "pow": "pom",
    # Hard manual word cases.
    "girf": "aw",
    "alalar": "stier",
    "fath3r": "faiher",
    "f@th3r": "faiher",
}


def canonical_alias(value: str) -> str:
    raw = str(value or "").strip().lower().strip(",. |'\"‘’“”_-")
    compact = clean(raw)
    return ALIASES.get(raw) or ALIASES.get(compact) or compact


def alias_similarity(a: str, b: str) -> float:
    ca = canonical_alias(a)
    cb = canonical_alias(b)
    if not ca or not cb:
        return 0.0
    return 1.0 if ca == cb else 0.0
