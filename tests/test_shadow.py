from __future__ import annotations

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.override import OverrideModel
from antibot_ai_ranker.shadow import ShadowDecision, build_shadow_decision, shadow_decision_to_json


def _ex() -> Example:
    return Example(
        case_id="case_001",
        attempt_id=1,
        source="manual_label",
        verdict="x",
        capture_path=None,  # type: ignore[arg-type]
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=["a", "b", "c"],
        solver_order=["a", "c", "b"],
    )


def test_build_shadow_decision_never_changes_production_order():
    decision = build_shadow_decision(
        _ex(),
        ai_order=["a", "b", "c"],
        ai_confidence=0.9,
        override_model=OverrideModel({"bias": 10.0}),
    )

    assert decision.production_order == ["a", "c", "b"]
    assert decision.shadow_order == ["a", "b", "c"]
    assert decision.would_override is True
    assert decision.mode == "shadow_no_submit"


def test_shadow_decision_json_contract_is_stable():
    payload = shadow_decision_to_json(
        ShadowDecision(
            case_id="case_001",
            mode="shadow_no_submit",
            production_order=["a"],
            shadow_order=["b"],
            ai_confidence=0.7,
            override_probability=0.8,
            would_override=True,
            family="words",
            source="manual_label",
        )
    )
    assert payload["schema_version"] == "antibot-ai-ranker.shadow.v1"
    assert payload["production_order"] == ["a"]
    assert payload["shadow_order"] == ["b"]
    assert payload["no_submit"] is True
