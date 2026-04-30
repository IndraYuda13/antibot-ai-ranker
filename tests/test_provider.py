from __future__ import annotations

from antibot_ai_ranker.provider import build_provider_decision


def test_build_provider_decision_reads_solver_shadow_payload():
    payload = {
        "schema_version": "antibot-image-solver.ranker-shadow.v1",
        "mode": "shadow_no_submit",
        "no_submit": True,
        "production_order": ["a", "b", "c"],
        "debug": {
            "instruction_ocr": ["ice, pan, tea"],
            "option_ocr": {"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        },
    }

    decision = build_provider_decision(payload)

    assert decision["provider"] == "antibot-ai-ranker"
    assert decision["no_submit"] is True
    assert "shadow_order" in decision
    assert "would_override" in decision
    assert "override_probability" in decision
    assert "ai_confidence" in decision


def test_build_provider_decision_falls_back_when_debug_missing():
    decision = build_provider_decision({"production_order": ["x", "y"]})

    assert decision["shadow_order"] == ["x", "y"]
    assert decision["would_override"] is False
    assert decision["status"] == "insufficient_debug_fallback"
