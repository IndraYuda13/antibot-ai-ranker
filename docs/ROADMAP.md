# Roadmap

## Phase 1: Baseline ranker

- Load live captures and labels from sibling projects.
- Train a stdlib-only pairwise ranker.
- Produce replay metrics and failure samples.
- Keep production solver untouched.

## Phase 2: Better offline science

Status: split-aware eval v1 implemented.

- Add train/validation split by attempt id to avoid memorizing recent patterns. ✅ v1 done
- Keep manual labels as optional held-out source. ✅ v1 done
- Mix synthetic JSONL with real data. ✅ v1 done
- Compare current rule solver vs AI ranker on the same cases. ✅ v1 done
- Add confidence calibration so low-confidence predictions can be flagged. ✅ threshold sweep v1 done; family threshold sweep v1 done; numeric feature pass done; alias feature pass done; dev-selected gate validation done; balanced manual calibration/test split done; safety objective v1 done; learned override gate v1 done; true out-of-distribution calibration still needs improvement.
- Add per-family error reports for numbers, short words, animal words, object words, and leetspeak.

## Phase 3: Synthetic data lab

Status: v1 implemented.

- Generate antibotlink-like question and option images.
- Support 3-option and 4-option layouts.
- Randomize font, color, rotation, blur, noise, compression, punctuation, and leetspeak.
- Export perfect labels automatically.
- Next: add stronger family coverage, train/dev/test split, and synthetic-real mix evaluation.

## Phase 4: Colab GPU lane

- Train a neural reranker or small vision model on synthetic + real data.
- Use Colab T4 only after the dataset generator and eval gates are stable.

## Phase 5: Production integration gate

- Integrate only if held-out manual labels improve over current solver.
- Require live post-restart soak window before promoting any model path.

- [x] Add multi-seed override validation to test split stability.
- [ ] Add more disagreement labels before live soak.

- [x] Add conservative override threshold calibration.
- [ ] Collect more accepted/raw disagreement negatives because calibration-only safety still misses heldout raw regressions.

- [x] Add disagreement mining/export for hard negative and positive override examples.
- [ ] Use mined disagreement rows to train a stronger source/family-aware override gate.

- [x] Add disagreement-trained override gate using mined positive/negative disagreement examples.
- [ ] Run lighter/faster multi-seed disagreement-gate sweep after optimizing report speed.

- [x] Add fast multi-seed disagreement-gate validation.
- [ ] Add shadow-mode export/contract for future production solver soak without submit.

- [x] Add shadow/no-submit JSON contract and export CLI.
- [ ] Wire future production solver only in no-submit shadow mode first, using the JSON contract.

- [x] Add `shadow-provider` stdin/stdout command compatible with `antibot-image-solver` external provider hook.
- [ ] Replace conservative default-weights provider with persisted trained disagreement-gate artifact before live soak.

- [x] Add train/export/load path for persisted disagreement-gate artifact.
- [in progress] Run live no-submit soak with `ANTIBOT_RANKER_SHADOW_PROVIDER` pointing at `shadow-provider --artifact ...`. Service env is installed in ClaimCoin, but first live collection is blocked because the only enabled account is on daily faucet limit with no `/faucet/verify` form.
