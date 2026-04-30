from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_shadow_export_writes_report(tmp_path: Path):
    output = tmp_path / "shadow.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "antibot_ai_ranker.cli",
            "shadow-export",
            "--epochs",
            "1",
            "--gate-epochs",
            "1",
            "--limit",
            "20",
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["output"] == str(output)
    saved = json.loads(output.read_text())
    assert saved["no_submit"] is True
    assert saved["schema_version"] == "antibot-ai-ranker.shadow-report.v1"
