from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from .dataset import Example
from .train import evaluate_examples, train_perceptron


@dataclass(frozen=True)
class DatasetSplit:
    train: list[Example]
    dev: list[Example]
    test: list[Example]
    heldout: list[Example]

    @property
    def all_examples(self) -> list[Example]:
        return self.train + self.dev + self.test + self.heldout


def split_examples(
    examples: list[Example],
    *,
    seed: int = 1337,
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    holdout_sources: set[str] | None = None,
) -> DatasetSplit:
    holdout_sources = holdout_sources or set()
    heldout = [ex for ex in examples if ex.source in holdout_sources]
    pool = [ex for ex in examples if ex.source not in holdout_sources]
    rng = random.Random(seed)
    shuffled = sorted(pool, key=lambda ex: ex.case_id)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_n = int(n * train_ratio)
    dev_n = int(n * dev_ratio)
    return DatasetSplit(
        train=shuffled[:train_n],
        dev=shuffled[train_n : train_n + dev_n],
        test=shuffled[train_n + dev_n :],
        heldout=sorted(heldout, key=lambda ex: ex.case_id),
    )


def _counts(split: DatasetSplit) -> dict[str, int]:
    return {
        "train": len(split.train),
        "dev": len(split.dev),
        "test": len(split.test),
        "heldout": len(split.heldout),
        "total": len(split.all_examples),
    }


def train_dev_test_report(
    examples: list[Example],
    *,
    seed: int = 1337,
    epochs: int = 8,
    holdout_sources: set[str] | None = None,
) -> dict[str, object]:
    split = split_examples(examples, seed=seed, holdout_sources=holdout_sources)
    weights = train_perceptron(split.train, epochs=epochs, seed=seed)
    return {
        "seed": seed,
        "epochs": epochs,
        "holdout_sources": sorted(holdout_sources or []),
        "counts": _counts(split),
        "train_metrics": evaluate_examples(split.train, weights),
        "dev_metrics": evaluate_examples(split.dev, weights),
        "test_metrics": evaluate_examples(split.test, weights),
        "heldout_metrics": evaluate_examples(split.heldout, weights) if split.heldout else {"total": 0, "ok": 0, "wrong": 0, "accuracy": 0.0, "by_source": {}, "failures": []},
        "weights": weights,
    }
