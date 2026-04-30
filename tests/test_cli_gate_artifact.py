from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_train_gate_artifact_writes_file(tmp_path: Path):
    output = tmp_path / "gate.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "antibot_ai_ranker.cli",
            "train-gate-artifact",
            "--epochs",
            "1",
            "--gate-epochs",
            "1",
            "--limit",
            "140",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["output"] == str(output)
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == "antibot-ai-ranker.disagreement-gate.v1"
