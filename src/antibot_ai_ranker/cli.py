from __future__ import annotations

import argparse
import json
from pathlib import Path

from .balanced_validation import balanced_manual_gate_report, safety_balanced_gate_report
from .multiseed_validation import multiseed_override_report
from .override_validation import conservative_override_gate_report, override_gate_report
from .benchmark import benchmark_orders
from .confidence import sweep_family_thresholds, sweep_thresholds
from .dataset import dataset_summary, load_examples, load_examples_with_synthetic
from .disagreements import mine_disagreements, summarize_disagreements
from .features import predict_order, predict_order_scored
from .splits import train_dev_test_report
from .synthetic import SyntheticConfig, generate_dataset
from .train import default_weights, evaluate_examples, save_model, train_perceptron
from .validation import validated_family_gate_report


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
    bench = sub.add_parser("benchmark")
    bench.add_argument("--epochs", type=int, default=8)
    bench.add_argument("--threshold", type=float, default=0.9)
    bench.add_argument("--limit", type=int)
    bench.add_argument("--source", action="append", default=[])
    calib = sub.add_parser("calibrate")
    calib.add_argument("--epochs", type=int, default=8)
    calib.add_argument("--limit", type=int)
    calib.add_argument("--source", action="append", default=[])
    calib_family = sub.add_parser("calibrate-family")
    calib_family.add_argument("--epochs", type=int, default=8)
    calib_family.add_argument("--limit", type=int)
    calib_family.add_argument("--source", action="append", default=[])
    gate = sub.add_parser("validate-gate")
    gate.add_argument("--epochs", type=int, default=8)
    gate.add_argument("--seed", type=int, default=1337)
    gate.add_argument("--limit", type=int)
    gate.add_argument("--holdout-source", action="append", default=[])
    balanced = sub.add_parser("validate-balanced")
    balanced.add_argument("--epochs", type=int, default=8)
    balanced.add_argument("--seed", type=int, default=1337)
    balanced.add_argument("--limit", type=int)
    balanced.add_argument("--manual-calibration-ratio", type=float, default=0.5)
    safety = sub.add_parser("validate-safety")
    safety.add_argument("--epochs", type=int, default=8)
    safety.add_argument("--seed", type=int, default=1337)
    safety.add_argument("--limit", type=int)
    safety.add_argument("--manual-calibration-ratio", type=float, default=0.5)
    safety.add_argument("--accepted-penalty", type=float, default=10.0)
    override = sub.add_parser("validate-override")
    override.add_argument("--epochs", type=int, default=8)
    override.add_argument("--seed", type=int, default=1337)
    override.add_argument("--limit", type=int)
    override.add_argument("--manual-calibration-ratio", type=float, default=0.5)
    override.add_argument("--override-epochs", type=int, default=25)
    conservative = sub.add_parser("validate-conservative")
    conservative.add_argument("--epochs", type=int, default=8)
    conservative.add_argument("--seed", type=int, default=1337)
    conservative.add_argument("--limit", type=int)
    conservative.add_argument("--manual-calibration-ratio", type=float, default=0.5)
    conservative.add_argument("--override-epochs", type=int, default=25)
    conservative.add_argument("--min-accepted-accuracy", type=float, default=100.0)
    multiseed = sub.add_parser("validate-multiseed")
    multiseed.add_argument("--epochs", type=int, default=8)
    multiseed.add_argument("--limit", type=int)
    multiseed.add_argument("--manual-calibration-ratio", type=float, default=0.5)
    multiseed.add_argument("--override-epochs", type=int, default=25)
    multiseed.add_argument("--seeds", default="11,22,33,44,55")
    multiseed.add_argument("--conservative", action="store_true")
    multiseed.add_argument("--min-accepted-accuracy", type=float, default=100.0)
    mine = sub.add_parser("mine-disagreements")
    mine.add_argument("--epochs", type=int, default=8)
    mine.add_argument("--seed", type=int, default=1337)
    mine.add_argument("--limit", type=int)
    mine.add_argument("--output", default="artifacts/disagreements.jsonl")
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

    if args.cmd == "mine-disagreements":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        weights = train_perceptron(examples, epochs=args.epochs, seed=args.seed)
        predictions: dict[str, list[str]] = {}
        confidences: dict[str, float] = {}
        for ex in examples:
            pred = predict_order_scored(ex, weights)
            predictions[ex.case_id] = pred.order
            confidences[ex.case_id] = pred.confidence
        rows = mine_disagreements(examples, predictions, confidences)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"output": str(output), "summary": summarize_disagreements(rows)}, indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-conservative":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        print(json.dumps(conservative_override_gate_report(
            examples,
            epochs=args.epochs,
            seed=args.seed,
            manual_calibration_ratio=args.manual_calibration_ratio,
            override_epochs=args.override_epochs,
            min_accepted_accuracy=args.min_accepted_accuracy,
        ), indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-multiseed":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
        print(json.dumps(multiseed_override_report(
            examples,
            seeds=seeds,
            epochs=args.epochs,
            manual_calibration_ratio=args.manual_calibration_ratio,
            override_epochs=args.override_epochs,
            conservative=args.conservative,
            min_accepted_accuracy=args.min_accepted_accuracy,
        ), indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-override":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        print(json.dumps(override_gate_report(
            examples,
            epochs=args.epochs,
            seed=args.seed,
            manual_calibration_ratio=args.manual_calibration_ratio,
            override_epochs=args.override_epochs,
        ), indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-safety":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        print(json.dumps(safety_balanced_gate_report(
            examples,
            epochs=args.epochs,
            seed=args.seed,
            manual_calibration_ratio=args.manual_calibration_ratio,
            accepted_penalty=args.accepted_penalty,
        ), indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-balanced":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        print(json.dumps(balanced_manual_gate_report(
            examples,
            epochs=args.epochs,
            seed=args.seed,
            manual_calibration_ratio=args.manual_calibration_ratio,
        ), indent=2, ensure_ascii=False))
        return

    if args.cmd == "validate-gate":
        examples = load_examples()
        if args.limit:
            examples = examples[: args.limit]
        report = validated_family_gate_report(
            examples,
            epochs=args.epochs,
            seed=args.seed,
            holdout_sources=set(args.holdout_source or []),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.cmd == "calibrate-family":
        examples = load_examples()
        if args.source:
            allowed = set(args.source)
            examples = [ex for ex in examples if ex.source in allowed]
        if args.limit:
            examples = examples[: args.limit]
        weights = train_perceptron(examples, epochs=args.epochs)
        scored = {ex.case_id: predict_order_scored(ex, weights) for ex in examples}
        predictions = {case_id: pred.order for case_id, pred in scored.items()}
        confidences = {case_id: pred.confidence for case_id, pred in scored.items()}
        print(json.dumps(sweep_family_thresholds(examples, predictions, confidences), indent=2, ensure_ascii=False))
        return

    if args.cmd == "calibrate":
        examples = load_examples()
        if args.source:
            allowed = set(args.source)
            examples = [ex for ex in examples if ex.source in allowed]
        if args.limit:
            examples = examples[: args.limit]
        weights = train_perceptron(examples, epochs=args.epochs)
        scored = {ex.case_id: predict_order_scored(ex, weights) for ex in examples}
        predictions = {case_id: pred.order for case_id, pred in scored.items()}
        confidences = {case_id: pred.confidence for case_id, pred in scored.items()}
        print(json.dumps(sweep_thresholds(examples, predictions, confidences), indent=2, ensure_ascii=False))
        return

    if args.cmd == "benchmark":
        examples = load_examples()
        if args.source:
            allowed = set(args.source)
            examples = [ex for ex in examples if ex.source in allowed]
        if args.limit:
            examples = examples[: args.limit]
        weights = train_perceptron(examples, epochs=args.epochs)
        predictions = {ex.case_id: predict_order(ex, weights) for ex in examples}
        # Current baseline ranker has no calibrated confidence yet. Use 1.0 so
        # hybrid shows the maximum possible impact of trusting AI disagreements.
        confidences = {ex.case_id: 1.0 for ex in examples}
        print(json.dumps(benchmark_orders(examples, predictions, ai_confidences=confidences, threshold=args.threshold), indent=2, ensure_ascii=False))
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
