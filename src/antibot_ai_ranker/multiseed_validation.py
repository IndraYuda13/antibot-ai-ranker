from __future__ import annotations

from collections.abc import Iterable

from .dataset import Example
from .override_validation import conservative_override_gate_report, override_gate_report


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 2),
        "mean": round(sum(values) / len(values), 2),
        "max": round(max(values), 2),
    }


def _accuracy(run: dict[str, object], section: str, lane: str) -> float:
    return float(((run.get(section) or {}).get(lane) or {}).get("accuracy", 0.0))  # type: ignore[union-attr]


def summarize_multiseed(runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "real_hybrid_accuracy": _stat([_accuracy(run, "real_test_metrics", "hybrid") for run in runs]),
        "manual_hybrid_accuracy": _stat([_accuracy(run, "manual_test_metrics", "hybrid") for run in runs]),
        "real_ai_accuracy": _stat([_accuracy(run, "real_test_metrics", "ai") for run in runs]),
        "manual_ai_accuracy": _stat([_accuracy(run, "manual_test_metrics", "ai") for run in runs]),
        "real_rule_accuracy": _stat([_accuracy(run, "real_test_metrics", "rule") for run in runs]),
        "manual_rule_accuracy": _stat([_accuracy(run, "manual_test_metrics", "rule") for run in runs]),
        "override_training_examples": _stat([float(run.get("override_training_examples", 0)) for run in runs]),
    }


def multiseed_override_report(
    examples: list[Example],
    *,
    seeds: Iterable[int],
    epochs: int = 8,
    manual_calibration_ratio: float = 0.5,
    override_epochs: int = 25,
    conservative: bool = False,
    min_accepted_accuracy: float = 100.0,
) -> dict[str, object]:
    seed_list = list(seeds)
    report_fn = conservative_override_gate_report if conservative else override_gate_report
    runs = [
        report_fn(
            examples,
            epochs=epochs,
            seed=seed,
            manual_calibration_ratio=manual_calibration_ratio,
            override_epochs=override_epochs,
            **({"min_accepted_accuracy": min_accepted_accuracy} if conservative else {}),
        )
        for seed in seed_list
    ]
    return {
        "seeds": seed_list,
        "epochs": epochs,
        "manual_calibration_ratio": manual_calibration_ratio,
        "override_epochs": override_epochs,
        "conservative": conservative,
        "min_accepted_accuracy": min_accepted_accuracy if conservative else None,
        "runs": runs,
        "summary": summarize_multiseed(runs),
    }
