from __future__ import annotations

import json
import subprocess
import sys


def test_cli_shadow_provider_reads_stdin_json():
    payload = {
        "production_order": ["a", "b", "c"],
        "debug": {
            "instruction_ocr": ["ice, pan, tea"],
            "option_ocr": {"a": ["1c3"], "b": ["p@n"], "c": ["t3@"]},
        },
    }
    proc = subprocess.run(
        [sys.executable, "-m", "antibot_ai_ranker.cli", "shadow-provider"],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    decision = json.loads(proc.stdout)
    assert decision["provider"] == "antibot-ai-ranker"
    assert decision["no_submit"] is True
    assert "shadow_order" in decision
