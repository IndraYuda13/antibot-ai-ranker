from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import dataset_summary, load_examples, load_examples_with_synthetic
from .splits import train_dev_test_report
from .synthetic import SyntheticConfig, generate_dataset
from .train import default_weights, evaluate_examples, save_model, train_perceptron


def main() -> None:
    parser = argparse.ArgumentParser(prog="antibot-ranker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    train = sub.add_parser("train")
    train.add_argument("--output", default="artifacts/model.json")
    train.add_argument("--epochs", type=int, default=8)
    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--model", default="artifacts/model.json")
    synth = sub.add_parser("generate-synthetic")
    synth.add_argument("--count", type=int, required=True)
    synth.add_argument("--options", type=int, default=3)
    synth.add_argument("--output-dir", default="data/synthetic")
    synth.add_argument("--seed", type=int, default=1337)
    synth.add_argument("--no-noise", action="store_true")
    synth.add_argument("--dark-theme", action="store_true")
    split_eval = sub.add_parser("split-eval")
    split_eval.add_argument("--epochs", type=int, default=8)
    split_eval.add_argument("--seed", type=int, default=1337)
    split_eval.add_argument("--holdout-source", action="append", default=[])
    split_eval.add_argument("--synthetic-jsonl", action="append", default=[])
    split_eval.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()

    if args.cmd == "summary":
        print(json.dumps(dataset_summary(), indent=2, ensure_ascii=False))
        return

    if args.cmd == "split-eval":
        examples = load_examples_with_synthetic(
            [Path(p) for p in args.synthetic_jsonl],
            include_real=not args.synthetic_only,
        )
        report = train_dev_test_report(
            examples,
            seed=args.seed,
            epochs=args.epochs,
            holdout_sources=set(args.holdout_source or []),
        )
        # Do not dump learned weights by default; keep CLI output readable.
        report.pop("weights", None)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.cmd == "generate-synthetic":
        cfg = SyntheticConfig(
            option_count=args.options,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            noise=not args.no_noise,
            dark_theme=args.dark_theme,
        )
        path = generate_dataset(count=args.count, cfg=cfg)
        print(json.dumps({"path": str(path), "count": args.count, "option_count": args.options}, indent=2, ensure_ascii=False))
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
        return


if __name__ == "__main__":
    main()
