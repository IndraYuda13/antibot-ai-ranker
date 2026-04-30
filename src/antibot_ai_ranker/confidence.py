from __future__ import annotations

from collections.abc import Iterable, Mapping

from .benchmark import benchmark_orders
from .dataset import Example
from .family import example_family


def order_confidence(*, best_score: float, second_best_score: float) -> float:
    if best_score <= 0:
        return 0.0
    gap = max(0.0, best_score - second_best_score) / max(abs(best_score), 1e-9)
    strength = min(1.0, abs(best_score) / 10.0)
    return round(max(0.0, min(1.0, 0.65 * gap + 0.35 * strength)), 4)


def sweep_thresholds(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    values = list(thresholds if thresholds is not None else [i / 20 for i in range(0, 21)])
    rows = []
    best = None
    for threshold in values:
        report = benchmark_orders(examples, ai_predictions, ai_confidences=confidences, threshold=threshold)
        compact = {
            "threshold": threshold,
            "rule": report["rule"],
            "ai": report["ai"],
            "hybrid": report["hybrid"],
            "by_source": report["by_source"],
        }
        rows.append(compact)
        if best is None or compact["hybrid"]["ok"] > best["hybrid"]["ok"] or (
            compact["hybrid"]["ok"] == best["hybrid"]["ok"] and compact["threshold"] > best["threshold"]
        ):
            best = compact
    return {"best": best or {}, "thresholds": rows}


def sweep_family_thresholds(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
    *,
    thresholds: Iterable[float] | None = None,
) -> dict[str, object]:
    grouped: dict[str, list[Example]] = {}
    for ex in examples:
        grouped.setdefault(example_family(ex), []).append(ex)
    families: dict[str, object] = {}
    best_thresholds: dict[str, float] = {}
    for family, items in grouped.items():
        report = sweep_thresholds(items, ai_predictions, confidences, thresholds=thresholds)
        families[family] = report
        best_thresholds[family] = float(report.get("best", {}).get("threshold", 1.0))
    return {"best_thresholds": best_thresholds, "families": families}
