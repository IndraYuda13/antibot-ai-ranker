from __future__ import annotations

from collections import Counter
from typing import Mapping

from .dataset import Example
from .family import example_family


def mine_disagreements(
    examples: list[Example],
    ai_predictions: Mapping[str, list[str]],
    confidences: Mapping[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ex in examples:
        rule_order = list(ex.solver_order or [])
        ai_order = list(ai_predictions.get(ex.case_id) or [])
        expected_order = list(ex.expected_order)
        if not rule_order or not ai_order or rule_order == ai_order:
            continue
        rule_ok = rule_order == expected_order
        ai_ok = ai_order == expected_order
        if rule_ok and not ai_ok:
            category = "negative_rule_ok_ai_wrong"
        elif not rule_ok and ai_ok:
            category = "positive_rule_wrong_ai_ok"
        elif not rule_ok and not ai_ok:
            category = "both_wrong"
        else:
            category = "both_ok_different_order"
        rows.append(
            {
                "case_id": ex.case_id,
                "attempt_id": ex.attempt_id,
                "source": ex.source,
                "family": example_family(ex),
                "category": category,
                "confidence": float(confidences.get(ex.case_id, 0.0)),
                "rule_ok": rule_ok,
                "ai_ok": ai_ok,
                "expected_order": expected_order,
                "solver_order": rule_order,
                "ai_order": ai_order,
                "question_ocr": ex.question_ocr,
                "option_ocr": ex.option_ocr,
                "capture_path": str(ex.capture_path) if ex.capture_path else None,
            }
        )
    return rows


def summarize_disagreements(rows: list[Mapping[str, object]]) -> dict[str, object]:
    by_category = Counter(str(row.get("category")) for row in rows)
    by_source = Counter(str(row.get("source")) for row in rows)
    by_family = Counter(str(row.get("family")) for row in rows)
    return {
        "total": len(rows),
        "by_category": dict(sorted(by_category.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_family": dict(sorted(by_family.items())),
    }
