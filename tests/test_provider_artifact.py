from __future__ import annotations

from pathlib import Path

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.gate_artifact import export_gate_artifact
from antibot_ai_ranker.provider import build_provider_decision


def _ex(i: int, source: str, expected: list[str], rule: list[str]) -> Example:
    return Example(
        case_id=f"{source}_{i:03d}",
        attempt_id=i,
        source=source,
        verdict="x",
        capture_path=Path(""),
        question_ocr=["ice, pan, tea"],
        option_ocr={"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        expected_order=expected,
        solver_order=rule,
    )


def test_provider_loads_gate_artifact(tmp_path: Path):
    examples = [_ex(i, "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(40)]
    examples += [_ex(i, "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(20)]
    artifact = tmp_path / "gate.json"
    export_gate_artifact(examples, artifact, epochs=1, gate_epochs=1, seed=4)
    payload = {
        "production_order": ["a", "c", "b"],
        "debug": {
            "instruction_ocr": ["ice, pan, tea"],
            "option_ocr": {"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        },
    }

    decision = build_provider_decision(payload, artifact_path=artifact)

    assert decision["provider"] == "antibot-ai-ranker"
    assert decision["artifact_schema_version"] == "antibot-ai-ranker.disagreement-gate.v1"
    assert decision["status"] == "artifact_gate_decision_logged"
