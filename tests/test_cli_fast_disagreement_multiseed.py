from __future__ import annotations

import json
import subprocess
import sys


def test_cli_validate_disagreement_multiseed_outputs_summary():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "antibot_ai_ranker.cli",
            "validate-disagreement-multiseed",
            "--epochs",
            "1",
            "--gate-epochs",
            "1",
            "--limit",
            "120",
            "--seeds",
            "1,2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["seeds"] == [1, 2]
    assert "summary" in payload
