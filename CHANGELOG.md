# Changelog

## 2026-04-30

- Created standalone ClaimCoin AntiBot AI ranker research repo.
- Added Superpowers design spec.
- Added dataset loader for ClaimCoin SQLite attempts, capture JSON, accepted-success weak labels, and manual labels.
- Added stdlib-only feature ranker baseline with CLI commands:
  - `claimcoin-ranker summary`
  - `claimcoin-ranker train`
  - `claimcoin-ranker evaluate`
- Added tests for text normalization, path resolution, and simple rank prediction.
- First local baseline reached about `95.23%` replay accuracy on current imported examples. This is research-only and not a production claim.
