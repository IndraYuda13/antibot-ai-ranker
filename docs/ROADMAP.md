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
- Add confidence calibration so low-confidence predictions can be flagged. ✅ threshold sweep v1 done; family threshold sweep v1 done; numeric feature pass done; alias feature pass done; dev-selected gate validation done; balanced manual calibration/test split done; true calibrated confidence still needs improvement.
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
