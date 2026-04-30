from __future__ import annotations

from dataclasses import dataclass

from .dataset import Example
from .family import example_family
from .override import OverrideModel, override_features, override_probability, predict_override

SCHEMA_VERSION = "antibot-ai-ranker.shadow.v1"


@dataclass(frozen=True)
class ShadowDecision:
    case_id: str
    mode: str
    production_order: list[str]
    shadow_order: list[str]
    ai_confidence: float
    override_probability: float
    would_override: bool
    family: str
    source: str


def build_shadow_decision(
    example: Example,
    *,
    ai_order: list[str],
    ai_confidence: float,
    override_model: OverrideModel,
) -> ShadowDecision:
    production_order = list(example.solver_order or [])
    features = override_features(example, ai_order, ai_confidence)
    probability = override_probability(override_model, features)
    would_override = predict_override(override_model, features)
    return ShadowDecision(
        case_id=example.case_id,
        mode="shadow_no_submit",
        production_order=production_order,
        shadow_order=list(ai_order) if would_override else production_order,
        ai_confidence=round(float(ai_confidence), 6),
        override_probability=round(float(probability), 6),
        would_override=would_override,
        family=example_family(example),
        source=example.source,
    )


def shadow_decision_to_json(decision: ShadowDecision) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": decision.mode,
        "no_submit": True,
        "case_id": decision.case_id,
        "source": decision.source,
        "family": decision.family,
        "production_order": decision.production_order,
        "shadow_order": decision.shadow_order,
        "would_override": decision.would_override,
        "ai_confidence": decision.ai_confidence,
        "override_probability": decision.override_probability,
    }
