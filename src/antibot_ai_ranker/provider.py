from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Example
from .features import predict_order_scored
from .gate_artifact import load_gate_artifact
from .override import OverrideModel, override_features, override_probability as score_override_probability, predict_override
from .train import default_weights

PROVIDER_NAME = "antibot-ai-ranker"


def _example_from_solver_payload(payload: dict[str, Any]) -> Example | None:
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    question = [str(x) for x in (debug.get("instruction_ocr") or [])]
    option_ocr = {str(k): [str(x) for x in (v or [])] for k, v in (debug.get("option_ocr") or {}).items()}
    production_order = [str(x) for x in (payload.get("production_order") or [])]
    if not question or not option_ocr or not production_order:
        return None
    return Example(
        case_id=str(payload.get("request_id") or "shadow_provider"),
        attempt_id=0,
        source=str(payload.get("source") or "shadow_provider"),
        verdict="shadow_no_submit",
        capture_path=Path(""),
        question_ocr=question,
        option_ocr=option_ocr,
        expected_order=production_order,
        solver_order=production_order,
    )


def _fallback(payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    production_order = [str(x) for x in (payload.get("production_order") or [])]
    return {
        "provider": PROVIDER_NAME,
        "status": status,
        "no_submit": True,
        "shadow_order": production_order,
        "would_override": False,
        "override_probability": 0.0,
        "ai_confidence": 0.0,
    }


def build_provider_decision(payload: dict[str, Any], *, artifact_path: str | Path | None = None) -> dict[str, Any]:
    example = _example_from_solver_payload(payload)
    if example is None:
        return _fallback(payload, status="insufficient_debug_fallback")

    production_order = list(example.solver_order or [])
    if artifact_path:
        artifact = load_gate_artifact(artifact_path)
        prediction = predict_order_scored(example, artifact.ranker_weights)
        model = OverrideModel(artifact.override_weights, artifact.override_threshold)
        features = override_features(example, prediction.order, prediction.confidence)
        probability = score_override_probability(model, features)
        would_override = bool(prediction.order and prediction.order != production_order and predict_override(model, features))
        return {
            "provider": PROVIDER_NAME,
            "status": "artifact_gate_decision_logged",
            "artifact_schema_version": artifact.schema_version,
            "no_submit": True,
            "shadow_order": prediction.order if would_override else production_order,
            "would_override": would_override,
            "override_probability": round(float(probability), 6),
            "ai_confidence": round(float(prediction.confidence), 6),
        }

    prediction = predict_order_scored(example, default_weights())
    would_override = bool(prediction.order and prediction.order != production_order)
    override_probability = prediction.confidence if would_override else 0.0
    return {
        "provider": PROVIDER_NAME,
        "status": "ranker_decision_logged",
        "no_submit": True,
        "shadow_order": prediction.order if would_override else production_order,
        "would_override": would_override,
        "override_probability": round(float(override_probability), 6),
        "ai_confidence": round(float(prediction.confidence), 6),
    }
