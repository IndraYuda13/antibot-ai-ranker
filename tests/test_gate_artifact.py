from __future__ import annotations

from pathlib import Path

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.gate_artifact import export_gate_artifact, load_gate_artifact


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


def test_export_and_load_gate_artifact_roundtrip(tmp_path: Path):
    examples = [_ex(i, "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(40)]
    examples += [_ex(i, "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(20)]
    output = tmp_path / "gate.json"

    payload = export_gate_artifact(examples, output, epochs=1, gate_epochs=1, seed=9)
    loaded = load_gate_artifact(output)

    assert payload["schema_version"] == "antibot-ai-ranker.disagreement-gate.v1"
    assert loaded.schema_version == "antibot-ai-ranker.disagreement-gate.v1"
    assert loaded.ranker_weights
    assert loaded.override_weights
    assert loaded.metadata["seed"] == 9
