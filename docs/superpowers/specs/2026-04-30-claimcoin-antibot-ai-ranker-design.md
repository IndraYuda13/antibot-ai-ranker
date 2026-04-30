# ClaimCoin AntiBot AI Ranker Design

## Goal
Build a separate research repo for a ClaimCoin AntiBot ranking model. The current production solver repo stays untouched except for future explicit integration. This project learns from existing live captures and manual labels, then predicts the most likely correct option order from OCR/debug features.

## Scope
Initial scope is a feature-based ranker, not a full vision model. The first model consumes OCR candidates, normalized forms, matcher scores, token/option similarities, and label/accepted-success ground truth. The target is to rerank existing solver outputs and identify low-confidence cases before production use.

## Data Sources
- ClaimCoin SQLite attempts from `projects/claimcoin-autoclaim/state/claimcoin.sqlite3`.
- Raw capture JSON files referenced by the DB.
- Manual labels from `projects/antibot-image-solver/state/antibot-labeling/labeled`.
- Accepted-success attempts can be used as weak ground truth because the website accepted the submitted order.

## Architecture
1. `dataset.py` loads captures and labels into canonical training examples.
2. `features.py` converts every token-option pair into numeric features.
3. `train.py` trains a lightweight baseline ranker using only Python stdlib first, with optional future ML backends.
4. `evaluate.py` compares baseline ranker accuracy against stored solver orders.
5. `synthetic.py` is a later lane for generated antibotlink-like datasets, including 3-option and 4-option cases.

## Model Strategy
Start with a transparent pairwise scorer and perceptron-style baseline. This is enough to prove whether the data contains learnable signal before moving to XGBoost, LightGBM, PyTorch, or a Colab T4 vision model. GPU training is deferred until the data generator and evaluation gates are stable.

## Success Gates
- Imports current live data without modifying source projects.
- Builds a dataset summary with accepted raw, manual labels, option counts, and split counts.
- Runs tests offline.
- Provides a clear README explaining why this is not yet production and how it can evolve toward synthetic + Colab training.

## Non-goals
- No production service replacement yet.
- No direct changes to `antibot-image-solver`.
- No claim of 100% live winrate from the current dataset.
