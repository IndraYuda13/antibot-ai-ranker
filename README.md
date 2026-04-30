# Antibot AI Ranker

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Status](https://img.shields.io/badge/status-research-orange) ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Research repo for learning better option ordering on antibotlink-style AntiBot image challenges.

This project is intentionally separate from the live solver. It imports existing captures and labels, builds a dataset, and trains a transparent baseline ranker first. The goal is to prove whether the current data can improve live ordering before moving to larger models or Google Colab GPU training.

## Why this exists

The current rule-based solver can replay labeled data cleanly, but live traffic keeps producing new OCR edge cases. A ranker can learn from accepted attempts and manual labels, then help choose the most likely option order or flag low-confidence cases.

## Current strategy

1. Import antibot attempts and capture JSON from a source adapter.
2. Merge manual labels when present.
3. Treat accepted-success attempts as weak ground truth.
4. Extract token-option similarity features.
5. Train a small transparent baseline.
6. Evaluate before any production integration.

## Data status at creation

At creation time, the live data source had roughly:

- 1.2k raw attempts
- 1.1k accepted-success weak labels
- 87 manual labels
- 3-option captures only

That is useful for a first ranker, but not enough to honestly promise 100% live winrate. The planned next lane is synthetic generation for 3-option and 4-option variants.

## First baseline

Current stdlib-only ranker baseline on the available local data:

- Total imported examples: `1201`
- Replay OK: `1150`
- Wrong: `51`
- Accuracy: `95.75%`

This is a baseline, not the final model. It proves the project can load the data, train a simple ranker, and produce measurable eval output. The repo also includes a synthetic AntiBotLinks-style generator for fast dataset expansion.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . pytest
antibot-ranker summary
antibot-ranker generate-synthetic --count 10000 --options 3 --output-dir data/synthetic-3opt
antibot-ranker generate-synthetic --count 10000 --options 4 --output-dir data/synthetic-4opt
antibot-ranker train --output artifacts/model.json
antibot-ranker evaluate --model artifacts/model.json
antibot-ranker split-eval --epochs 4 --holdout-source manual_label
antibot-ranker benchmark --epochs 4
antibot-ranker calibrate --epochs 4
antibot-ranker calibrate-family --epochs 4
pytest
```

Sample outputs are stored in [`examples/`](examples/). Split-aware evaluation is available through `antibot-ranker split-eval`; rule-vs-AI comparison is available through `antibot-ranker benchmark`, and global threshold sweep is available through `antibot-ranker calibrate`; family-aware threshold sweep is available through `antibot-ranker calibrate-family`.

Default paths assume this repo lives at:

`/root/.openclaw/workspace/projects/antibot-ai-ranker`

and source data is configured through environment variables if the default local adapter is not present:

- `ANTIBOT_RANKER_SOURCE_ROOT`
- `ANTIBOT_RANKER_SOLVER_ROOT`
- `ANTIBOT_RANKER_DB_NAME`
- `ANTIBOT_RANKER_CASE_PREFIX`

## Roadmap and findings

- Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Research findings: [`docs/FINDINGS.md`](docs/FINDINGS.md)

## Boundary

This is research tooling. It does not replace the live solver yet. No model from this repo should be integrated into production until it beats the current rule solver on held-out manual labels and then survives a live post-restart soak window.
