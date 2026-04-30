# Changelog

## 2026-04-30

- Renamed project branding from target-specific naming to universal `Antibot AI Ranker`.
- Created standalone Antibot AI ranker research repo.
- Added Superpowers design spec.
- Added dataset loader for Antibot SQLite attempts, capture JSON, accepted-success weak labels, and manual labels.
- Added stdlib-only feature ranker baseline with CLI commands:
  - `antibot-ranker summary`
  - `antibot-ranker train`
  - `antibot-ranker evaluate`
- Added tests for text normalization, path resolution, simple rank prediction, synthetic generation, synthetic loader, and synthetic CLI.
- First local baseline reached about `95.75%` replay accuracy on current imported examples after refresh.
- Added synthetic AntiBotLinks-style dataset generator v1 with 3-option and 4-option support.
- Added split-aware train/dev/test evaluation with optional manual-label holdout and synthetic JSONL mixing.
- Added rule-vs-AI-vs-hybrid benchmark CLI and sample outputs.
- Added confidence/threshold sweep CLI and sample calibration outputs.
- Added family-aware challenge classifier and family threshold calibration CLI.
- Added numeric-aware features for digits, number words, roman numerals, simple math expressions, and leet number-word forms.
- Added OCR alias features for short-word and leetspeak hard cases.
- Added dev-selected family gate validation CLI to expose calibration behavior on test/heldout splits.
- Added balanced manual calibration/test validation CLI for honest hard-case gate evaluation.
- Added safety-objective gate validation with accepted-raw regression penalty.
- Added learned override-gate validation to decide when AI may replace rule order. This is research-only and not a production claim.

- Added multi-seed override validation to measure split stability.

- Added conservative override threshold calibration and multi-seed conservative mode.

- Added disagreement mining/export CLI for rule-vs-AI gate training data.

- Added disagreement-trained override gate validation using mined positive/negative examples.

- Added fast multi-seed disagreement-gate validation and recorded stable five-seed results.

- Added shadow/no-submit JSON contract and `shadow-export` CLI for safe future soak testing.
