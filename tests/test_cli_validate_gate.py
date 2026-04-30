from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_gate_outputs_test_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "validate-gate", "--epochs", "1", "--limit", "100"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "selected_thresholds" in payload
    assert "test_metrics" in payload
    assert payload["counts"]["test"] > 0
