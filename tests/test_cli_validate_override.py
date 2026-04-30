from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_override_outputs_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "validate-override", "--epochs", "1", "--limit", "160"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "override_training_examples" in payload
    assert "manual_test_metrics" in payload
    assert "real_test_metrics" in payload
