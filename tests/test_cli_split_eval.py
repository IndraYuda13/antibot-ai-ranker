from __future__ import annotations

import json
import subprocess
import sys


def test_cli_split_eval_returns_counts_and_holdout_metrics():
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "split-eval", "--epochs", "1", "--holdout-source", "manual_label"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["counts"]["train"] > 0
    assert payload["counts"]["heldout"] >= 0
    assert "test_metrics" in payload
    assert "heldout_metrics" in payload
