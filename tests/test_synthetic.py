from __future__ import annotations

import json

from antibot_ai_ranker.synthetic import SyntheticConfig, generate_dataset, generate_example


def test_generate_example_preserves_solution_order_and_shuffle(tmp_path):
    cfg = SyntheticConfig(option_count=4, seed=123, output_dir=tmp_path)
    example = generate_example("synthetic_000001", cfg)

    assert example["challenge_id"] == "synthetic_000001"
    assert example["option_count"] == 4
    assert len(example["correct_order"]) == 4
    assert set(example["correct_order"]) == set(example["option_texts"])
    assert example["images"]["question"].endswith("question.png")
    assert (tmp_path / example["images"]["question"]).exists()
    for rel in example["images"]["options"].values():
        assert (tmp_path / rel).exists()


def test_generate_dataset_writes_jsonl_and_images(tmp_path):
    cfg = SyntheticConfig(option_count=3, seed=7, output_dir=tmp_path)
    jsonl = generate_dataset(count=5, cfg=cfg)

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(rows) == 5
    assert all(row["option_count"] == 3 for row in rows)
    assert all(len(row["correct_order"]) == 3 for row in rows)
    assert all((tmp_path / row["images"]["question"]).exists() for row in rows)
