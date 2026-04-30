from __future__ import annotations

import json
import subprocess
import sys


def test_cli_calibrate_outputs_best_threshold():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "calibrate", "--epochs", "1", "--limit", "100"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "best" in payload
    assert "thresholds" in payload
    assert payload["best"]["hybrid"]["total"] > 0
