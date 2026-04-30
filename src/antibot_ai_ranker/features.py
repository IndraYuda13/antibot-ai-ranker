from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from difflib import SequenceMatcher

from .dataset import Example
from .textnorm import clean, similarity


@dataclass(frozen=True)
class PairFeatures:
    token: str
    option_id: str
    values: dict[str, float]


@dataclass(frozen=True)
class OrderPrediction:
    order: list[str]
    confidence: float
    best_score: float
    second_best_score: float


def token_sets(question_ocr: list[str], option_count: int) -> list[list[str]]:
    out: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate in question_ocr:
        text = str(candidate).replace("\n", " ").strip()
        for comma in (True, False):
            parts = [p.strip() for p in (text.split(",") if comma else text.split()) if p.strip()]
            if len(parts) == option_count:
                key = tuple(parts)
                if key not in seen:
                    seen.add(key)
                    out.append(parts)
    return out


def pair_features(token: str, option_candidates: list[str]) -> dict[str, float]:
    top = option_candidates[0] if option_candidates else ""
    sims = [similarity(token, cand) for cand in option_candidates[:5]]
    top_clean = clean(top)
    token_clean = clean(token)
    return {
        "bias": 1.0,
        "top_similarity": sims[0] if sims else 0.0,
        "best_similarity": max(sims) if sims else 0.0,
        "mean_similarity": sum(sims) / len(sims) if sims else 0.0,
        "exact_clean": 1.0 if token_clean and token_clean == top_clean else 0.0,
        "token_len": float(len(token_clean)),
        "top_len": float(len(top_clean)),
        "len_delta": float(abs(len(token_clean) - len(top_clean))),
        "seq_ratio": SequenceMatcher(None, token_clean, top_clean).ratio() if token_clean and top_clean else 0.0,
        "candidate_count": float(len(option_candidates)),
    }


def score_pair(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in features.items())


def _confidence(best_score: float, second_best_score: float) -> float:
    if best_score <= 0:
        return 0.0
    gap = max(0.0, best_score - second_best_score) / max(abs(best_score), 1e-9)
    strength = min(1.0, abs(best_score) / 10.0)
    return round(max(0.0, min(1.0, 0.65 * gap + 0.35 * strength)), 4)


def predict_order_scored(example: Example, weights: dict[str, float]) -> OrderPrediction:
    best_order: list[str] = []
    best_score = float("-inf")
    second_best_score = float("-inf")
    option_ids = list(example.option_ocr.keys())
    for tokens in token_sets(example.question_ocr, len(option_ids)):
        if len(tokens) != len(option_ids):
            continue
        for perm in permutations(option_ids, len(tokens)):
            score = 0.0
            for token, option_id in zip(tokens, perm):
                score += score_pair(weights, pair_features(token, example.option_ocr[option_id]))
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_order = list(perm)
            elif score > second_best_score:
                second_best_score = score
    if second_best_score == float("-inf"):
        second_best_score = 0.0
    if not best_order:
        best_order = option_ids
        best_score = 0.0
    return OrderPrediction(best_order, _confidence(best_score, second_best_score), best_score, second_best_score)


def predict_order(example: Example, weights: dict[str, float]) -> list[str]:
    return predict_order_scored(example, weights).order
