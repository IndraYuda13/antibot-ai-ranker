# ClaimCoin AntiBot AI Ranker

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Status](https://img.shields.io/badge/status-research-orange) ![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Research repo for learning better option ordering on ClaimCoin-style AntiBot image challenges.

This project is intentionally separate from the live solver. It imports existing captures and labels, builds a dataset, and trains a transparent baseline ranker first. The goal is to prove whether the current data can improve live ordering before moving to larger models or Google Colab GPU training.

## Why this exists

The current rule-based solver can replay labeled data cleanly, but live traffic keeps producing new OCR edge cases. A ranker can learn from accepted attempts and manual labels, then help choose the most likely option order or flag low-confidence cases.

## Current strategy

1. Import ClaimCoin attempts and capture JSON.
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

- Total imported examples: `1195`
- Replay OK: `1138`
- Wrong: `57`
- Accuracy: `95.23%`

This is a baseline, not the final model. It proves the project can load the data, train a simple ranker, and produce measurable eval output.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . pytest
claimcoin-ranker summary
claimcoin-ranker train --output artifacts/model.json
claimcoin-ranker evaluate --model artifacts/model.json
pytest
```

Sample outputs are stored in [`examples/`](examples/).

Default paths assume this repo lives at:

`/root/.openclaw/workspace/projects/claimcoin-antibot-ai-ranker`

and source data lives in sibling projects:

- `../claimcoin-autoclaim`
- `../antibot-image-solver`

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Boundary

This is research tooling. It does not replace the live solver yet. No model from this repo should be integrated into production until it beats the current rule solver on held-out manual labels and then survives a live post-restart soak window.
