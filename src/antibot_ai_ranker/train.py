from __future__ import annotations

import json
import random
from pathlib import Path

from .dataset import Example, load_examples
from .features import pair_features, predict_order, score_pair, token_sets


def default_weights() -> dict[str, float]:
    return {
        "bias": 0.0,
        "top_similarity": 1.0,
        "best_similarity": 1.2,
        "mean_similarity": 0.2,
        "exact_clean": 2.0,
        "numeric_match": 4.0,
        "numeric_mismatch": -2.0,
        "token_has_number": 0.0,
        "top_has_number": 0.0,
        "token_len": 0.0,
        "top_len": 0.0,
        "len_delta": -0.05,
        "seq_ratio": 1.0,
        "candidate_count": 0.0,
    }


def train_perceptron(examples: list[Example], *, epochs: int = 8, seed: int = 1337) -> dict[str, float]:
    weights = default_weights()
    rng = random.Random(seed)
    train = list(examples)
    for _ in range(max(1, epochs)):
        rng.shuffle(train)
        for ex in train:
            option_ids = set(ex.option_ocr)
            if set(ex.expected_order) != option_ids:
                continue
            pred = predict_order(ex, weights)
            if pred == ex.expected_order:
                continue
            token_options = token_sets(ex.question_ocr, len(option_ids))
            if not token_options:
                continue
            tokens = token_options[0]
            for token, good_id, bad_id in zip(tokens, ex.expected_order, pred):
                good = pair_features(token, ex.option_ocr[good_id])
                bad = pair_features(token, ex.option_ocr[bad_id])
                for name in set(good) | set(bad):
                    weights[name] = weights.get(name, 0.0) + 0.1 * (good.get(name, 0.0) - bad.get(name, 0.0))
    return weights


def evaluate_examples(examples: list[Example], weights: dict[str, float]) -> dict[str, object]:
    total = ok = 0
    by_source: dict[str, dict[str, int]] = {}
    failures = []
    for ex in examples:
        pred = predict_order(ex, weights)
        passed = pred == ex.expected_order
        total += 1
        ok += int(passed)
        bucket = by_source.setdefault(ex.source, {"total": 0, "ok": 0})
        bucket["total"] += 1
        bucket["ok"] += int(passed)
        if not passed and len(failures) < 30:
            failures.append({"case_id": ex.case_id, "source": ex.source, "expected": ex.expected_order, "predicted": pred, "question_ocr": ex.question_ocr[:3], "option_ocr": ex.option_ocr})
    return {"total": total, "ok": ok, "wrong": total - ok, "accuracy": round(ok / total * 100, 2) if total else 0.0, "by_source": by_source, "failures": failures}


def save_model(path: Path, weights: dict[str, float], metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_type": "stdlib_perceptron_ranker", "weights": weights, "metrics": metrics}, indent=2, ensure_ascii=False))
