# Model improvement notes

Based on the latest behavioral report (`report/08-28-12:41.md`), the latest
tensorboard run
(`DATA: mtl 64 tl 500 | MODEL: es 8  cs 32 nl 2 ee 8 ns 16 | TRAIN: lr 0.005 bs 64 gc 1.0 wd 1e-05 | TIME: 2026-08-28 12:39:12`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json`,
`emoji_keywords.py` and `test_model.py`.

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | 6/8 (75%) | `neutral` → Angry, `love` → Excited. Regressed from 8/8 in the previous pass. Love has 190 / 13,537 rows; the feeling head is also past its eval optimum (see §2.4). |
| Negations | 3/8 avoided the negated feeling, 1/6 hit the expected opposite | Only `not calm` → Anxious actually works. `not neutral` / `not love` "pass" only because they land on Angry. No representation of negation or word order. |
| Emoji keywords | 37/247 top-1 (15%), 68/247 top-3 (28%), 82/247 top-5 (33%), across 50 emojis | Regressed from 53/250 top-1 (21%) / 86/250 top-3 (34%). The switch to an 8-d contrastive emoji head + sampled softmax + `EMBED_SIZE` 16→8 made emoji worse. |

**Emoji classes that work** (≥60% top-1 in the report): 😊 (100%), 😌 (80%),
💔 / 😔 / 😭 (60%). All high-frequency and tied to one unambiguous lexical cue.

**Emoji classes at ~0% top-1**: 😤 😩 😅 😣 🔥 🙄 😒 😂 😕 😵‍💫 👻 🎶 💪 ⚡ 🧘 🥳 🙌
🌙 ☕(1/5). Two clusters fail: (a) object / nature / abstract emojis whose cue is
a noun several words in, (b) near-synonym negative faces (😣/😩/😞/😔, 😒/🙄).

**Attractor classes**: wrong predictions collapse onto a small set —
😊 🎉 😌 😤 ✅ 👻 🍣 💻 🌿 🙏. The model falls back on these whenever the input
lacks a short n-gram it has memorised. `🍣`, `💻`, `🚆` turning up as top-1 for
`sobbing` / `thinking` / `drained` is the long-tail-noise symptom from §2.2.

## 2. Root causes

### 2.1 The emoji head is an 8-dimensional contrastive bottleneck (biggest emoji problem)
`model.py`'s emoji head is CLIP-style: the 32-d max-pooled conv features go
through `text_proj = Linear(32 → 8)`, get L2-normalised, and are scored by
cosine similarity against 133 L2-normalised vectors from
`emoji_embedding = Embedding(133, 8)`, scaled by `exp(logit_scale)`
(`logit_scale` a learnable scalar, init 2.6593). `EMOJI_EMBED_SIZE = 8`.

Packing 133 classes onto the unit 7-sphere with any usable margin is
geometrically hopeless, and near-synonym emojis (😔/😞/😢/😭, 😠/😡/🤬) are forced
to nearly the same point. This is a **regression** from the previous plain
`Linear(→133)` full-softmax head: emoji top-1 dropped 21% → 15%, top-3 34% → 28%.

### 2.2 Sampled softmax never teaches the model to suppress the long tail
`train.py:sample_negatives` trains the emoji head as a **17-way** problem (the
true emoji + `NEGATIVE_SAMPLES = 16` random negatives per row, positive always in
column 0), but `evaluate()` and inference score the **full 133-way** softmax.
`train/emoji_loss` falls 3.27 → 1.32 (min 1.32 @195; ln 17 = 2.83, so the model
fits the sampled task well) while `eval/emoji_loss` sits at 3.53–3.76 the entire
run (min 3.525 @160; ln 133 = 4.89, i.e. only ~28% below chance). With median 60
rows/class and 87 of 133 classes under 80 rows, a rare emoji is seldom a positive
and seldom sampled as a negative, so its 8-d vector is barely trained and can
sit anywhere on the sphere — including directly under a common query direction.
That is exactly the attractor behaviour in the report.

### 2.3 Receptive field is ~5 characters
`model.py` conv stack: `Conv1d(k=3, padding=1)` × `NUM_LAYERS` (= 2), no dilation,
no stride, then a masked max-pool over time. Effective receptive field =
1 + 2·(3−1) = **5 chars** — under one word — and the max-pool discards position.
Consequences match the report:
- Negation is unrepresentable: "not happy" and "happy" share every 5-gram except
  "not ".
- One-word cues work ("angry", "coffee", "sad"); multi-word phrases
  ("crushing it", "my heart is full", "calm and peaceful") don't.

### 2.4 The feeling head overfits, and the checkpoint is selected on a noisy sum
Same tensorboard run: `train/feeling_loss` 1.85 → 0.88 (min 0.87 @197) while
`eval/feeling_loss` bottoms at **1.124 @ epoch 20** then drifts up to 1.244 @200
— textbook overfitting. `train.py` ships the checkpoint with the lowest
`eval_emoji_loss + eval_feeling_loss`; that sum is minimised at epoch ~30
(4.72), but it is dominated by `eval_emoji_loss`, whose entire spread over the
run is 0.23 and is indistinguishable from noise. Selection is therefore
effectively random with respect to the only term that moves (feeling), and it
was luck that it landed near the feeling optimum rather than at epoch 120–180.
The previous notes flagged *accuracy*-based selection; the code now selects on
loss, but on a sum where the informative term is drowned out.

### 2.5 133 emojis, ~two-thirds of them starved
`labels.json` holds **133** emojis (CLAUDE.md still says an "80-emoji palette" —
stale, along with its `main.py` / "LSTM" / `pack_padded_sequence` / `ExportWrapper`
references; the entry point is now `train.py`, the model is a char-CNN, and the
ONNX export is a plain trace with no wrapper). In `data.jsonl` (13,537 rows):
max 😤 596, **median 60**, min 😑 8 (then 🎟️ 13, 🥶 14, 🪑 15, 😏 17). **87 of
133 classes have < 80 rows, 96 have < 100** — unlearnable, and each still takes
softmax mass at eval. Many survivors are mutual near-synonyms (😔/😞/😢/😭,
😠/😡/🤬, 😄/😊/😃/😁/🙂), so top-1 is ambiguous even for a perfect model; top-3
(28%) is the fairer figure and it is still low.

### 2.6 Class imbalance, no compensation
Feelings: Anxious 2757, Happy 2533, Sad 1975, Angry 1902, Excited 1486,
Neutral 1390, Calm 1304, **Love 190** (14.5× under Anxious). `train.py` puts no
`weight=` on either `CrossEntropyLoss`, no `label_smoothing`, and
`train_data_loader` has no `WeightedRandomSampler`. `love` → Excited in the
report is the direct symptom; the emoji tail has the same problem, unweighted.

### 2.7 Eval split is 500 random rows and silently non-stationary
`TEST_LEN = 500` (~3.7% of the corpus). Across 133 emoji classes that is < 4
rows/class with many classes absent — which is why `eval/emoji_loss` is the flat
noise band above. `data.py:split()` does `random.Random(42).shuffle(data)` over
an **append-only** `data.jsonl`, so every `bun run gen_data.ts` reshuffles which
rows are held out and runs from before/after a data add are not comparable.
There is no de-duplication before the split, so near-identical texts from one
generation batch can straddle it (mild leak).

### 2.8 Minor
- `CHARS` in `data.py` still has no digits (`0-9` dropped by `normalize`);
  `VOCAB_SIZE = 38`.
- `normalize` now also collapses long character runs
  (`re.sub(r'(.)\1{2,}', r'\1\1', text)`: "soooo" → "soo"). `docs/app.js`
  currently mirrors this (`.replace(/(.)\1{2,}/g, "$1$1")`) — keep them
  byte-identical on any future change.
- `WEIGHT_DECAY = 1e-5` is effectively off; no dropout; 200 flat epochs at
  `LR = 5e-3` with no schedule.
- Train texts average 33 chars (full sentences); the battery feeds 1–3 word
  keywords. A word-level path would bridge the mismatch; the 5-gram model can't.

## 3. Recommendations (prioritised)

### Done since the previous notes
- `TEST_LEN` raised 200 → 500 (still far too small for 133 classes — see D).
- Checkpoint selection moved off a noisy accuracy proxy onto eval loss (but onto
  the wrong sum — see §2.4 and D).
- `normalize` run-collapse added and mirrored in `docs/app.js`.

### High impact

**A. Fix the emoji head — this is the regression.** Cheapest first:
1. Revert to a plain `Linear(CHANNELS → 133)` full-softmax cross-entropy head.
   The previous notes' 21% / 34% came from this; the contrastive rework lost it.
2. If the contrastive head is kept for its own sake: raise `EMOJI_EMBED_SIZE` to
   ≥ 64 **and** drop `sample_negatives` — with only 133 classes, full InfoNCE
   over all emojis costs nothing and removes the train/eval mismatch in §2.2.
3. Either way, add inverse-frequency class weights (or focal loss) on the emoji
   loss so the tail and the near-synonyms stop being trained purely as noise.

**B. Give the model context.** Pick one, cheapest first:
1. Dilated conv stack: keep kernel 3, dilations 1-2-4-8 (~5 layers) so the
   receptive field covers the full 64-char window.
2. Concatenate **masked mean-pool + masked max-pool** instead of max only —
   keeps frequency/energy information and reduces attractor collapse.
3. Conv + bidirectional GRU (CRNN): conv layers stay a learned n-gram extractor,
   the bi-RNN adds unbounded order-sensitive context. Pool the GRU outputs
   (masked mean ⧺ max), don't gather the last state — that keeps the ONNX export
   a plain trace. onnxruntime-web supports bidirectional GRU at opset 18. This
   is the option that can actually represent negation.

**C. Fix the label space.**
- Drop or fold emojis with < ~100 training rows; target 40–60 classes each with
  ≥ 100 rows (CLAUDE.md's intended "80" is a sane ceiling). 87 classes are
  currently under 80.
- Define explicit synonym groups (sad-faces, angry-faces, laughing-faces) and
  additionally score top-1 *within group*; 😔 vs 😞 is not recoverable from text.
- Reconcile `labels.json` (133) and CLAUDE.md (80 / LSTM / `main.py`).

**D. Make eval honest, and select on it.**
- Raise `TEST_LEN` to ~2000 (~15%).
- Make the split deterministic per row (hash of normalised text), so it is
  stable as `data.jsonl` grows; de-duplicate near-identical texts first.
- Report per-class emoji accuracy and **macro-F1** and top-3, not just micro
  top-1.
- Select the checkpoint on `feeling_loss` + an emoji metric that isn't noise
  (macro-F1 or top-3), and fold the `emoji_keywords` battery into the selection
  metric. As-is, the epoch-20 feeling optimum is invisible to selection.

**E. Rebalance data and sampling.**
- Use `add-samples` / `bun run gen_data.ts` but target the **tail**: generate for
  under-100 emojis and for Love / Calm / Neutral specifically, not random pairs.
- `WeightedRandomSampler` by inverse emoji frequency in `train_data_loader`,
  and/or inverse-freq `weight=` on both `CrossEntropyLoss` heads.

### Medium impact

**F. Loss / regularisation.**
- Weight the two task losses (e.g. `0.5·emoji + 1.0·feeling`, or uncertainty
  weighting) so the emoji loss stops dominating the shared trunk.
- `WEIGHT_DECAY` → ~1e-4; add dropout 0.1–0.2 after the conv stack once capacity
  grows.
- Add `label_smoothing` to the feeling head too.

**G. Capacity.** `EMBED_SIZE` 8 → 16 or 32, `CHANNELS` 32 → 64. Still < 1M
params; the headroom is in the trunk and emoji head, not the feeling head.

**H. Negation coverage** (only after B.3). Add training texts with negations
("not happy", "wasn't excited", "no longer sad") mapped to sensible labels; keep
the negation battery as a release gate.

### Low impact / hygiene

- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt`).
- LR schedule: cosine decay or `ReduceLROnPlateau`.
- Evaluate every epoch (not every 10) once the eval set is larger.
- `torch.manual_seed(0)` is set; also seed any DataLoader workers / set
  deterministic algorithms for fully comparable runs.

## 4. Suggested order of work

1. **D** — bigger deterministic eval + macro-F1/top-3 + fix the selection
   metric. Nothing else is measurable until this is done; the feeling head is
   already overfitting past epoch 20 and the current metric can't see it.
2. **A** — revert (or properly widen) the emoji head. This is the single change
   that recovers the 21% → 15% top-1 regression.
3. **C** — prune the emoji tail to ~50–60 classes with ≥ 100 rows.
4. **B.1 / B.2** — dilated convs + mean⧺max pool: one small `model.py` change
   that should move multi-word cues and reduce attractor collapse.
5. **E** (rebalance + sampler) and **F** (loss weighting / regularisation),
   then **G** (capacity).
6. **B.3** (conv + bi-GRU) and **H** if negation or top-3 is still short.
