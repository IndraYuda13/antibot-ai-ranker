from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from antibot_ai_ranker.dataset import Example
from antibot_ai_ranker.gate_artifact import export_gate_artifact


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


def test_cli_shadow_provider_loads_artifact(tmp_path: Path):
    examples = [_ex(i, "accepted_success_raw", ["a", "b", "c"], ["a", "b", "c"]) for i in range(40)]
    examples += [_ex(i, "manual_label", ["a", "b", "c"], ["a", "c", "b"]) for i in range(20)]
    artifact = tmp_path / "gate.json"
    export_gate_artifact(examples, artifact, epochs=1, gate_epochs=1, seed=5)
    payload = {
        "production_order": ["a", "c", "b"],
        "debug": {"instruction_ocr": ["ice, pan, tea"], "option_ocr": {"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]}},
    }
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "shadow-provider", "--artifact", str(artifact)],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    decision = json.loads(proc.stdout)
    assert decision["artifact_schema_version"] == "antibot-ai-ranker.disagreement-gate.v1"
