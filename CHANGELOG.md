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
- Added rule-vs-AI-vs-hybrid benchmark CLI and sample outputs. This is research-only and not a production claim.
