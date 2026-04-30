from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_mine_disagreements_writes_jsonl(tmp_path: Path):
    output = tmp_path / "rows.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "antibot_ai_ranker.cli",
            "mine-disagreements",
            "--epochs",
            "1",
            "--limit",
            "180",
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
    assert "summary" in payload
    assert output.exists()
