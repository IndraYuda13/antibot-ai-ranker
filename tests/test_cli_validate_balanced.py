from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_balanced_outputs_manual_test_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "validate-balanced", "--epochs", "1", "--limit", "140", "--manual-calibration-ratio", "0.5"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "selected_thresholds" in payload
    assert "manual_test_metrics" in payload
    assert "real_test_metrics" in payload
