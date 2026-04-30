from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import dataset_summary, load_examples
from .train import default_weights, evaluate_examples, save_model, train_perceptron


def main() -> None:
    parser = argparse.ArgumentParser(prog="claimcoin-ranker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    train = sub.add_parser("train")
    train.add_argument("--output", default="artifacts/model.json")
    train.add_argument("--epochs", type=int, default=8)
    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--model", default="artifacts/model.json")
    args = parser.parse_args()

    if args.cmd == "summary":
        print(json.dumps(dataset_summary(), indent=2, ensure_ascii=False))
        return

    examples = load_examples()
    if args.cmd == "train":
        weights = train_perceptron(examples, epochs=args.epochs)
        metrics = evaluate_examples(examples, weights)
        save_model(Path(args.output), weights, metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    if args.cmd == "evaluate":
        path = Path(args.model)
        weights = json.loads(path.read_text())["weights"] if path.exists() else default_weights()
        print(json.dumps(evaluate_examples(examples, weights), indent=2, ensure_ascii=False))
