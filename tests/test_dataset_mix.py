from __future__ import annotations

from antibot_ai_ranker.dataset import load_examples_with_synthetic
from antibot_ai_ranker.synthetic import SyntheticConfig, generate_dataset


def test_load_examples_with_synthetic_appends_generated_examples(tmp_path):
    jsonl = generate_dataset(count=3, cfg=SyntheticConfig(option_count=3, output_dir=tmp_path, seed=11))
    examples = load_examples_with_synthetic([jsonl], include_real=False)

    assert len(examples) == 3
    assert {ex.source for ex in examples} == {"synthetic_antibotlinks_v1"}
