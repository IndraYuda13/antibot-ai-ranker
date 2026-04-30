from __future__ import annotations

import json
import subprocess
import sys


def test_cli_calibrate_family_outputs_family_thresholds():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "calibrate-family", "--epochs", "1", "--limit", "120"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "best_thresholds" in payload
    assert "families" in payload
