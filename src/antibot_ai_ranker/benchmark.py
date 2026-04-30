from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from .dataset import Example


def choose_hybrid_order(rule_order: list[str], ai_order: list[str], *, ai_confidence: float, threshold: float = 0.9) -> list[str]:
    if ai_order and ai_confidence >= threshold and ai_order != rule_order:
        return list(ai_order)
    return list(rule_order)


def _empty_metrics() -> dict[str, object]:
    return {"total": 0, "ok": 0, "wrong": 0, "accuracy": 0.0}


def _add(metrics: dict[str, object], passed: bool) -> None:
    metrics["total"] = int(metrics["total"]) + 1
    metrics["ok"] = int(metrics["ok"]) + int(passed)
    metrics["wrong"] = int(metrics["total"]) - int(metrics["ok"])
    metrics["accuracy"] = round(int(metrics["ok"]) / int(metrics["total"]) * 100, 2) if int(metrics["total"]) else 0.0


def benchmark_orders(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    *,
    ai_confidences: Mapping[str, float] | None = None,
    threshold: float = 0.9,
) -> dict[str, object]:
    ai_confidences = ai_confidences or {}
    totals = {"rule": _empty_metrics(), "ai": _empty_metrics(), "hybrid": _empty_metrics()}
    by_source: dict[str, dict[str, dict[str, object]]] = defaultdict(lambda: {"rule": _empty_metrics(), "ai": _empty_metrics(), "hybrid": _empty_metrics()})
    disagreements = []
    for ex in examples:
        expected = list(ex.expected_order)
        rule_order = list(ex.solver_order or []) or list(ex.option_ocr.keys())
        ai_order = list(ai_predictions.get(ex.case_id) or [])
        confidence = float(ai_confidences.get(ex.case_id, 1.0 if ai_order else 0.0))
        hybrid_order = choose_hybrid_order(rule_order, ai_order, ai_confidence=confidence, threshold=threshold)
        orders = {"rule": rule_order, "ai": ai_order, "hybrid": hybrid_order}
        for name, order in orders.items():
            passed = bool(order) and order == expected
            _add(totals[name], passed)
            _add(by_source[ex.source][name], passed)
        if rule_order != ai_order and len(disagreements) < 50:
            disagreements.append({
                "case_id": ex.case_id,
                "source": ex.source,
                "expected": expected,
                "rule": rule_order,
                "ai": ai_order,
                "hybrid": hybrid_order,
                "ai_confidence": confidence,
            })
    return {**totals, "by_source": dict(by_source), "disagreements": disagreements}
