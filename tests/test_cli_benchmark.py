from __future__ import annotations

import json
import subprocess
import sys


def test_cli_benchmark_outputs_rule_ai_hybrid_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "benchmark", "--epochs", "1", "--limit", "80"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "rule" in payload
    assert "ai" in payload
    assert "hybrid" in payload
    assert payload["rule"]["total"] > 0
