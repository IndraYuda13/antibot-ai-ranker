# Findings

## 2026-04-30: path toward near-100% antibot ordering

### Practical conclusion

The best path is a hybrid system, not an immediate full replacement of the current solver.

Recommended runtime chain:

1. Existing solver extracts challenge images.
2. OCR layer reads question and option images with Tesseract / other OCR engines.
3. Rule matcher computes existing deterministic scores.
4. Antibot AI Ranker reranks candidate option orders from OCR/debug features.
5. Decision layer chooses:
   - AI order when confidence is high.
   - Current rule solver when AI confidence is low or model is out-of-distribution.
   - Label queue when live server rejects.

This lets the project improve accuracy without throwing away the working rule solver.

### Why not train full vision end-to-end first

A full image-to-order model is possible, but current real data is too small for a reliable production-grade vision model.

Current data is useful for a first ranker because the model learns from OCR/debug features. It is not enough to honestly claim permanent 100% live winrate.

Useful rough data targets:

- Feature ranker first pass: 1k-2k real examples. Current data is enough to start.
- Strong feature ranker: 5k-10k mixed real/synthetic examples.
- Better neural ranker: 20k-50k mixed examples plus held-out real labels.
- Vision model experiments: 20k-50k generated images minimum, 100k-300k better.
- Held-out real validation: keep at least 300-1000 manually checked real cases that are not used for training.

### Expected winrate realism

- 95-97% is realistic with ranker + current label loop.
- 98-99% likely needs synthetic generator + real hard-case validation.
- Permanent 100% is not a safe promise because target sites can keep producing unseen OCR variants. The practical target is near-100% with confidence gates, fallback, and continuous label/tune loop.

### AI integration options

#### Level 1: OCR + AI ranker

This is the recommended next implementation.

- Input: OCR candidates, normalized forms, matcher scores, token/option similarities, OCR candidate rank.
- Output: ordered option IDs plus confidence.
- Pros: works with current dataset, fast to train, easy to debug, low risk.
- Cons: still depends on OCR quality.

#### Level 2: small OCR/vision helper

Train a model to improve per-image text/category recognition.

- Pros: can reduce OCR edge cases.
- Cons: needs synthetic image dataset and careful real validation.

#### Level 3: end-to-end image-to-order model

Train one model from question/options images directly to final order.

- Pros: potentially strongest long term.
- Cons: highest data requirement, hardest to debug, easiest to overfit. Not the first production path.

### Synthetic dataset direction

Synthetic data is the fastest way to scale coverage. The generator should mimic the real antibotlink family instead of inventing unrelated captchas.

Minimum generated record format:

```json
{
  "challenge_id": "synthetic_000001",
  "option_count": 3,
  "question_text": "ice, pan, tea",
  "option_texts": {"a": "1c3", "b": "p@n", "c": "t3@"},
  "correct_order": ["a", "b", "c"],
  "images": {
    "question": "...base64 or path...",
    "options": {"a": "...", "b": "...", "c": "..."}
  },
  "metadata": {
    "font": "...",
    "angle": 0,
    "noise": true,
    "theme": "light"
  }
}
```

Dataset must include both 3-option and 4-option challenges from the start.

### AntiBotLinks open-source finding

There is an open-source PHP library that matches the target family closely:

- Repository: `https://github.com/miguilimzero/antibotlinks`
- Package: `miguilim/antibotlinks`
- License: MIT
- Summary from README: self-hosted image captcha library that uses a predefined dictionary and generates images that must be selected in a specific order.
- Packagist says it requires PHP `^8.0` and `intervention/image ^3.0`.

Observed implementation details from source:

- `generateLinks(int $amount = 4)` defaults to 4 options.
- It selects a word universe, samples/shuffles words, and sometimes flips key/value direction.
- It generates a phrase image from the ordered keys.
- It generates option images from values.
- It shuffles option display order after solution creation.
- It stores solution as ordered option IDs.
- `validateAnswer()` compares submitted concatenated IDs with stored solution.
- Image generation uses random fonts, random text angle, random colors, shadow text, and optional noise.

Important source URLs:

- README: `https://raw.githubusercontent.com/miguilimzero/antibotlinks/main/README.md`
- Core generator: `https://raw.githubusercontent.com/miguilimzero/antibotlinks/main/src/AntiBotLinks.php`
- Word universe/options: `https://raw.githubusercontent.com/miguilimzero/antibotlinks/main/src/Traits/HasOptions.php`

### Generator recommendation

Build a Python synthetic generator first, using AntiBotLinks behavior as the reference pattern. A PHP wrapper around the original library is also possible, but Python is easier to integrate with the ranker and Colab training.

Generator should support:

- Word universe families from AntiBotLinks.
- Custom faucet-like word families seen in live captures.
- 3 and 4 options.
- Key/value flip direction.
- Random option shuffle with preserved solution order.
- Fonts, rotation, color/shadow, noise, blur, compression.
- Export to JSONL plus image files.

### Next implementation steps

1. Add `synthetic.py` generator to this repo. ✅ v1 done.
2. Add `antibot-ranker generate-synthetic --count N --options 3|4`. ✅ v1 done.
3. Add dataset loader for synthetic JSONL. ✅ v1 done.
4. Add split-aware evaluation: train/dev/test by challenge ID and source type. ✅ v1 done.
5. Current split-eval smoke: real test `170/177` (`96.05%`), manual-label heldout `60/87` (`68.97%`), synthetic smoke test `32/60` (`53.33%`). These numbers show the baseline ranker is not strong enough yet for production; next step is better features/model and rule-solver comparison.
6. Add rule-solver vs AI-ranker benchmark on the same cases. ✅ v1 done. Current same-set benchmark: rule `1200/1269` (`94.56%`), AI `1216/1269` (`95.82%`). Manual labels only: rule `18/87` (`20.69%`), AI `59/87` (`67.82%`). Important caveat: this is not held-out proof yet; it shows the AI can help hard/manual cases but hurts some accepted-success raw cases. Threshold sweep v1 found best all-data hybrid at threshold `0.1`: hybrid `1226/1275` (`96.16%`), accepted raw `99.33%`, manual label `52.87%`. Manual-only still prefers trusting AI (`59/87`, `67.82%`). This means a single global confidence threshold is not enough; next step is source/family-aware confidence and better calibrated margins.
6. After synthetic v1 works, use Colab T4 for neural ranker experiments.

### Production gate

Do not integrate into `antibot-image-solver` until:

- Offline held-out real labels beat current solver.
- Synthetic-only improvements do not regress real hard cases.
- Confidence gate is calibrated.
- Live post-restart soak window improves compared to the rule-only baseline.


### Family-aware calibration v1

Family-aware threshold sweep now groups examples by question token family. Current all-data best thresholds: words `0.1`, short_words `0.05`, leetspeak `0.0`, animals `0.15`, number_words/numeric `1.0`. Manual-only results still show AI helps words/short/leetspeak, while numeric and number-word coverage remains weak. This confirms the next model work should improve numeric/number-word features and use family-aware gating rather than one global threshold.


### Numeric feature pass v1

Numeric-aware features now parse digits, number words, roman numerals, simple math expressions, and common leet number-word forms. This improved weak numeric families: manual-only numeric moved from `2/4` to `3/4`, number_words from `0/1` to `1/1`; all-data numeric moved from `4/6` to `5/6`, number_words from `8/9` to `9/9`. Short_words and leetspeak still need broader feature/model work, so the AI ranker is still research-only.


### OCR alias feature pass v1

Alias features now map common live OCR confusions such as `2p -> zip`, `200 -> zoo`, `20r -> zor`, `mc -> arc`, `cir -> or`, `teg -> 424`, `t03 -> te`, `mal -> mel`, and `dnt -> lem`. Manual-only family calibration improved words `9/11 -> 10/11` and short_words `40/57 -> 44/57` compared with the earlier family baseline. All-data words reached `244/248` (`98.39%`) and short_words `920/954` (`96.44%`). Numeric remains unstable after feature interactions, so future work needs a proper model/validation split rather than only weight tweaks.


### Dev-selected gate validation v1

`validate-gate` now selects family thresholds on the dev split and applies them to test/heldout. This exposed an important limitation: when manual labels are held out, dev contains mostly accepted-success raw cases, so selected thresholds become conservative (`1.0`) and the hybrid refuses AI overrides on hard manual labels. Manual heldout result: rule `18/87` (`20.69%`), AI `69/87` (`79.31%`), hybrid `18/87` (`20.69%`). Conclusion: confidence/gating cannot be learned only from accepted-success raw data. Next step should create a balanced calibration set with some manual hard cases in dev, while reserving a separate manual test set for honest proof.


### Balanced manual calibration/test v1

`validate-balanced` splits manual labels into calibration and honest manual test. With `manual_calibration_ratio=0.5`, calibration selected thresholds `{words:0.25, short_words:0.1, leetspeak:0.15, animals:1.0, number_words:1.0, numeric:1.0}`. Manual test improved from rule `9/44` (`20.45%`) to hybrid `35/44` (`79.55%`), while AI alone got `38/44` (`86.36%`). Real accepted test remained safest with rule `186/186`; hybrid was `179/186`. This confirms manual hard cases must be represented in calibration, but production integration still needs a gate that protects accepted-success raw better.


### Safety objective v1

`validate-safety` selects family thresholds with an explicit penalty for accepted-success raw regressions. With accepted penalty `10`, manual test hybrid reached `37/44` (`84.09%`) but real accepted test was only `179/187` (`95.72%`). With accepted penalty `25`, real accepted hybrid improved to `185/187` (`98.93%`) but manual test dropped to `18/44` (`40.91%`). This exposes the core tradeoff: current confidence is not sharp enough to protect accepted raw while capturing most manual hard-case gains. The next productive step is a better calibrated model/objective, not just threshold tweaking.


### Learned override gate v1

`validate-override` trains a small classifier on calibration disagreements to decide when the AI ranker may override the rule solver. With `manual_calibration_ratio=0.5` and `override_epochs=40`, it learned from `24` override examples. On the current live dataset snapshot, real accepted test stayed at `188/188` (`100%`) and manual hard test improved from rule `9/44` (`20.45%`) to hybrid `39/44` (`88.64%`). This is the first validation lane that preserves accepted raw while capturing almost all current manual hard-case gains. It is still not production-ready because the override classifier only has 24 supervised disagreement examples; next gate should expand disagreement data with synthetic and live soak samples, then repeat split/stability checks across seeds.


### Multi-seed learned override validation v1

`validate-multiseed --epochs 4 --override-epochs 40 --seeds 11,22,33,44,55` shows the learned override gate is promising but not yet production-safe. Across five seeds, real accepted hybrid accuracy was min `96.83%`, mean `99.37%`, max `100%`; manual hard hybrid accuracy was min `72.73%`, mean `78.64%`, max `84.09%`. Rule stayed `100%` on real accepted test and only mean `19.55%` on manual hard test. The override gate is a strong offline improvement over rule on hard cases, but seed 22 exposed accepted raw regression, so the next step is more disagreement data plus a stricter production gate before any solver integration.


### Conservative override calibration v1

`validate-conservative` calibrates the override model decision threshold on the calibration split with a minimum accepted/raw accuracy constraint. The single-seed result selected threshold `0.02`, with real accepted hybrid `187/189` (`98.94%`) and manual hard hybrid `39/44` (`88.64%`). Multi-seed conservative validation improved the real accepted minimum from `96.83%` to `97.35%`, mean `99.47%`, but still did not guarantee `100%` heldout accepted/raw. This proves calibration split safety alone is insufficient because the current calibration data does not contain enough accepted/raw disagreement negatives. Next work should collect more disagreement labels and add an explicit conservative fallback for unseen family/source patterns before production integration.


### Disagreement mining v1

`mine-disagreements --epochs 4` found `97` rule/AI disagreement rows: `39` negative cases where rule was correct and AI was wrong, `51` positive cases where rule was wrong and AI was correct, and `7` both-wrong cases. Source split was `39` accepted-success raw and `58` manual-label rows. Family split was dominated by `short_words` (`67` rows), followed by `leetspeak` (`14`) and `words` (`12`). This confirms the exact data needed for the next gate phase exists in the current corpus: accepted/raw negatives are no longer just theoretical, and should be used explicitly for conservative override training and evaluation.


### Disagreement-trained override gate v1

`validate-disagreement-gate --epochs 4 --gate-epochs 80 --negative-weight 8` trains the override gate directly from mined disagreement examples in the train+calibration pool. The run used `48` disagreement rows (`21` negative rule-correct/AI-wrong, `22` positive rule-wrong/AI-correct, `5` both-wrong) and `43` supervised override rows. Heldout real accepted test reached `205/205` (`100%`) for rule, AI, and hybrid; manual hard test improved from rule `9/44` (`20.45%`) to hybrid `38/44` (`86.36%`). This is the strongest single-split result so far because it uses the exact negative examples that earlier gates lacked. Multi-seed sweep still needs optimization because the naive repeated report was too slow and got killed; do not call this production-ready until that stability gate passes.


### Fast multi-seed disagreement-gate validation v1

`validate-disagreement-multiseed --epochs 4 --gate-epochs 80 --negative-weight 8 --seeds 11,22,33,44,55` completed without SIGKILL and is the first stability gate where heldout real accepted hybrid stayed `100%` across all five seeds. Summary: real accepted hybrid min/mean/max `100.0% / 100.0% / 100.0%`; manual hard hybrid min/mean/max `75.0% / 79.09% / 84.09%`; manual rule baseline mean only `19.55%`. Override training rows ranged `47-60` across seeds. This is strong enough to move to shadow-mode planning, but still not production integration: next gate should run the learned decision in a no-submit/live-shadow path and log disagreements before touching `antibot-image-solver` production behavior.


### Shadow/no-submit contract v1

`shadow-export` writes a stable `antibot-ai-ranker.shadow-report.v1` JSON report. Each decision uses `antibot-ai-ranker.shadow.v1`, includes `no_submit: true`, preserves `production_order`, records `shadow_order`, `would_override`, family/source, AI confidence, and override probability. Sample run `shadow-export --epochs 4 --gate-epochs 80 --negative-weight 8 --limit 25` wrote `25` accepted/raw decisions, all kept production order (`would_override=0`). This creates the safe contract for future live shadow/soak integration: production solver can log ranker decisions without changing submitted answers.
