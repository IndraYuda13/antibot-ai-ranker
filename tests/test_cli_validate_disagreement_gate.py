from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_disagreement_gate_outputs_training_summary():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "validate-disagreement-gate", "--epochs", "1", "--limit", "180", "--gate-epochs", "1"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "disagreement_training" in payload
    assert "real_test_metrics" in payload
    assert "manual_test_metrics" in payload
