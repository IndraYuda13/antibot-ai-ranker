from __future__ import annotations

from antibot_ai_ranker.dataset import load_synthetic_examples
from antibot_ai_ranker.synthetic import SyntheticConfig, generate_dataset


def test_load_synthetic_examples_from_generated_jsonl(tmp_path):
    jsonl = generate_dataset(count=2, cfg=SyntheticConfig(option_count=4, output_dir=tmp_path, seed=9))
    examples = load_synthetic_examples(jsonl)

    assert len(examples) == 2
    assert all(ex.source == "synthetic_antibotlinks_v1" for ex in examples)
    assert all(len(ex.expected_order) == 4 for ex in examples)
    assert all(len(ex.option_ocr) == 4 for ex in examples)
