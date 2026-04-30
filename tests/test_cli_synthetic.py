from __future__ import annotations

import json
import subprocess
import sys


def test_cli_generate_synthetic_writes_requested_dataset(tmp_path):
    out_dir = tmp_path / "synthetic"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "antibot_ai_ranker.cli",
            "generate-synthetic",
            "--count",
            "3",
            "--options",
            "4",
            "--output-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["count"] == 3
    assert payload["option_count"] == 4
    assert (out_dir / "synthetic_4opt_3.jsonl").exists()
