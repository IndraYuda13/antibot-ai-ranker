from __future__ import annotations

from collections.abc import Iterable

from .dataset import Example
from .disagreement_gate import train_disagreement_gate_report


def _stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {"min": round(min(values), 2), "mean": round(sum(values) / len(values), 2), "max": round(max(values), 2)}


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "real_hybrid_accuracy": _stat([float(run.get("real_hybrid_accuracy", 0.0)) for run in runs]),
        "manual_hybrid_accuracy": _stat([float(run.get("manual_hybrid_accuracy", 0.0)) for run in runs]),
        "real_rule_accuracy": _stat([float(run.get("real_rule_accuracy", 0.0)) for run in runs]),
        "manual_rule_accuracy": _stat([float(run.get("manual_rule_accuracy", 0.0)) for run in runs]),
        "real_ai_accuracy": _stat([float(run.get("real_ai_accuracy", 0.0)) for run in runs]),
        "manual_ai_accuracy": _stat([float(run.get("manual_ai_accuracy", 0.0)) for run in runs]),
        "override_training_examples": _stat([float(run.get("override_training_examples", 0.0)) for run in runs]),
    }


def _compact_run(report: dict[str, object]) -> dict[str, object]:
    real = report["real_test_metrics"]  # type: ignore[index]
    manual = report["manual_test_metrics"]  # type: ignore[index]
    disagreement_training = report.get("disagreement_training", {})
    return {
        "seed": report["seed"],
        "real_hybrid_accuracy": real["hybrid"]["accuracy"],  # type: ignore[index]
        "manual_hybrid_accuracy": manual["hybrid"]["accuracy"],  # type: ignore[index]
        "real_rule_accuracy": real["rule"]["accuracy"],  # type: ignore[index]
        "manual_rule_accuracy": manual["rule"]["accuracy"],  # type: ignore[index]
        "real_ai_accuracy": real["ai"]["accuracy"],  # type: ignore[index]
        "manual_ai_accuracy": manual["ai"]["accuracy"],  # type: ignore[index]
        "override_training_examples": report.get("override_training_examples", 0),
        "disagreement_training_total": disagreement_training.get("total", 0) if isinstance(disagreement_training, dict) else 0,
    }


def fast_disagreement_multiseed_report(
    examples: list[Example],
    *,
    seeds: Iterable[int],
    epochs: int = 4,
    gate_epochs: int = 80,
    negative_weight: float = 8.0,
    manual_calibration_ratio: float = 0.5,
) -> dict[str, object]:
    seed_list = list(seeds)
    runs = [
        _compact_run(
            train_disagreement_gate_report(
                examples,
                epochs=epochs,
                seed=seed,
                gate_epochs=gate_epochs,
                negative_weight=negative_weight,
                manual_calibration_ratio=manual_calibration_ratio,
            )
        )
        for seed in seed_list
    ]
    return {
        "seeds": seed_list,
        "epochs": epochs,
        "gate_epochs": gate_epochs,
        "negative_weight": negative_weight,
        "manual_calibration_ratio": manual_calibration_ratio,
        "runs": runs,
        "summary": summarize_runs(runs),
    }
