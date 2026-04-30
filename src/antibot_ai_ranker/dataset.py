from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import SourcePaths


@dataclass(frozen=True)
class Example:
    case_id: str
    attempt_id: int
    source: str
    verdict: str
    capture_path: Path
    question_ocr: list[str]
    option_ocr: dict[str, list[str]]
    expected_order: list[str]

    @property
    def option_ids(self) -> list[str]:
        return list(self.option_ocr.keys())


def _split_order(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return [part for part in str(value or "").split() if part]


def load_capture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _case_id(attempt_id: int, prefix: str) -> str:
    return f"{prefix}_{attempt_id:06d}"


def _manual_labels(paths: SourcePaths) -> dict[str, list[str]]:
    labeled_dir = paths.label_root / "labeled"
    labels: dict[str, list[str]] = {}
    if not labeled_dir.exists():
        return labels
    for item in labeled_dir.glob("*.json"):
        data = json.loads(item.read_text())
        order = _split_order((data.get("manual_label") or {}).get("correct_answer_order"))
        if order:
            labels[item.stem] = order
    return labels


def load_examples(paths: SourcePaths | None = None, *, include_weak: bool = True, include_manual: bool = True) -> list[Example]:
    paths = paths or SourcePaths()
    labels = _manual_labels(paths) if include_manual else {}
    examples: list[Example] = []
    con = sqlite3.connect(paths.db_path)
    rows = con.execute("select id, verdict, capture_path from antibot_attempts order by id asc").fetchall()
    for attempt_id_raw, verdict, capture_rel in rows:
        attempt_id = int(attempt_id_raw)
        cid = _case_id(attempt_id, paths.case_prefix)
        capture_path = paths.source_root / str(capture_rel)
        if not capture_path.exists():
            continue
        capture = load_capture(capture_path)
        debug = ((capture.get("solver") or {}).get("debug") or {})
        question = [str(x) for x in (debug.get("instruction_ocr") or [])]
        options = {str(k): [str(x) for x in (v or [])] for k, v in (debug.get("option_ocr") or {}).items()}
        if not question or not options:
            continue
        if cid in labels:
            expected = labels[cid]
            source = "manual_label"
        elif include_weak and verdict == "accepted_success":
            expected = _split_order((capture.get("solver") or {}).get("ordered_ids") or (capture.get("solver") or {}).get("antibotlinks"))
            source = "accepted_success_raw"
        else:
            continue
        if expected:
            examples.append(Example(cid, attempt_id, source, str(verdict), capture_path, question, options, expected))
    return examples


def dataset_summary(paths: SourcePaths | None = None) -> dict[str, Any]:
    paths = paths or SourcePaths()
    examples = load_examples(paths)
    by_source: dict[str, int] = {}
    option_counts: dict[int, int] = {}
    for ex in examples:
        by_source[ex.source] = by_source.get(ex.source, 0) + 1
        option_counts[len(ex.option_ocr)] = option_counts.get(len(ex.option_ocr), 0) + 1
    return {"total_examples": len(examples), "by_source": by_source, "option_counts": option_counts, "db_path": str(paths.db_path)}
