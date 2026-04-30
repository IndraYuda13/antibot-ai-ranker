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
