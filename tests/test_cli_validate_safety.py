from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_safety_outputs_manual_and_real_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "validate-safety", "--epochs", "1", "--limit", "150"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "selected_thresholds" in payload
    assert "real_test_metrics" in payload
    assert "manual_test_metrics" in payload
    assert "safety_selection" in payload
