# Model improvement notes

Based on the latest behavioral report (`report/08-28-10:43.md`) plus a read of
`model.py`, `train.py`, `data.py`, `config.py`, `labels.json` and the current
`data.jsonl`.

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | 8/8 (100%) | Trivial: the prompt *is* the label word. Only proves the feeling head isn't broken. |
| Negations | 1/8 avoided the negated feeling, 0/6 hit the opposite | The model has **no** representation of negation or word order. |
| Emoji keywords | 53/250 top-1 (21%), 86/250 top-3 (34%), 50 emojis | Weak, and lopsided (see below). |

**Emoji classes that work** (≥60% top-1): 😊 (100%), 😭 (80%), 😌/😠/😟/😡 (60%),
🌮 (60%). These are either high-frequency or map to a single unambiguous lexical
cue ("taco", "sobbing").

**Emoji classes at ~0%**: 🧘 🌿 🌙 🎶 😕 💪 ⚡ 🙄 🙌 😣 😅 🚀 🤫 🕊️ 😵‍💫 😒 😴(partial).
Two clusters fail: (a) abstract / object / nature emojis whose cue is a noun
several words into the sentence, (b) subtle negative-face distinctions
(😣/😩/😞/😔, 😒/🙄) that are near-synonyms.

**Attractor classes**: wrong predictions collapse onto a tiny set — 👀 📍 👍 ✅ 🌊 😌.
The model falls back on these whenever the input doesn't contain a strong short
n-gram it has memorised.

## 2. Root causes

### 2.1 Receptive field is ~7 characters (biggest problem)
`model.py` is a char-CNN: `Conv1d(k=3)` ×(1 + `NUM_LAYERS`=2) = 3 stacked
kernel-3 layers, no dilation, then a global max-pool over time. Effective
receptive field = 1 + 3·(3−1) = **7 chars ≈ one short word**. The classifier
therefore sees a bag of char-7-grams with their positions discarded by max-pool.

Consequences that match the report exactly:
- Negation is impossible to represent ("not happy" and "happy" share every
  7-gram except "not ").
- Single-word cues work ("angry", "coffee", "taco"); multi-word semantic
  phrases ("crushing it", "my heart is full", "calm and peaceful") don't.
- Uncertain inputs collapse onto whichever classes own the most common short
  n-grams → the 👀/📍/✅ attractors.

### 2.2 Capacity is tiny for a 133-way head
`EMBED_SIZE=16`, `H_SIZE=32` → ~13k total params. The emoji head is a 32→133
linear; 32 dims is a severe bottleneck for 133 fine-grained classes. The feeling
head (32→8) is not bottlenecked, which is why it saturates at 100%.

### 2.3 The label space blew up to 133 emojis with a brutal long tail
`labels.json` now holds **133** emojis (CLAUDE.md still says an "80-emoji
palette" — stale, along with its "LSTM" / `main.py` references). In `data.jsonl`
(12,188 rows): max class 504, **median 56**, min 8. Roughly 15 emojis have <25
samples (😑 🥶 😏 🪑 🎟️ 💸 📺 🚆 🍼 🎵 …) — unlearnable, and they still take
softmax mass. Many remaining classes are mutual near-synonyms
(😔/😞/😢/😭, 😠/😡/🤬, 😄/😊/😃/😁/🙂), so **top-1 is ambiguous even for a
perfect model** — top-3 (34%) is the fairer number and it's still low.

### 2.4 Label noise from the generation pipeline
`gen_data.ts` now generates generic topic/voice texts, then an LLM annotates
each with one *free-choice* "best-fit" emoji. Single annotator + free choice →
high variance, popularity bias toward common faces, and no guarantee the text
actually discriminates that emoji from a sibling. Training targets are
inconsistent and the top-1 metric is measured against a noisy gold label.

### 2.5 Class imbalance, no weighting
Feelings: Anxious 2397 … **Love 181** (13× under-represented), Neutral 1188.
Nothing in `train.py` compensates — `feeling_ce` has no `weight`, `emoji_ce`
only has `label_smoothing`. No `WeightedRandomSampler`.

### 2.6 Eval set is 200 random rows (~1.6%)
With 133 emoji classes that's <2 samples/class and many classes absent, so
`emoji_acc` on the held-out loader is very noisy — and `train.py` selects the
shipped checkpoint on `mean(emoji_acc, feeling_acc)` of that noisy number.
Worse: `split()` does `random.Random(42).shuffle(list)` over an **append-only**
file, so the eval set silently changes every time the corpus grows and runs
stop being comparable. Near-duplicate texts from the same generation batch can
also straddle the split (mild leak → eval is even weaker than it looks).

### 2.7 Minor
- No digits in `CHARS` (0–9 dropped by `normalize`).
- `WEIGHT_DECAY=1e-6` is effectively off; no dropout; 100 flat epochs at
  `LR=5e-3` with no schedule.
- Train texts average 31 chars (full sentences); the report battery feeds 1–3
  word keywords — a real word-level model would bridge this, the 7-gram model
  can't.

## 3. Recommendations (prioritised)

### High impact

**A. Give the model context.** Pick one, cheapest first:
1. Dilated conv stack: keep kernel 3, dilations 1‑2‑4‑8 (≈6 layers) so the
   receptive field covers the full 64-char window.
2. Concatenate **mean-pool + max-pool** instead of max-only — keeps
   frequency/energy info and reduces attractor collapse.
3. For negation and word order: a **conv + bi-GRU (CRNN)** — conv layers stay a
   learned n-gram extractor, the bi-RNN adds unbounded order-sensitive context.
   See §3.1 for a concrete spec. (A 2-layer Transformer encoder is the
   alternative, but at 12k rows / len 64 the BiGRU is the safer bet.)
4. Add a word-level path: hash-embed whitespace-split tokens and concat with the
   char features. "peace", "zen", "yoga", "music", "moon" would then map
   directly instead of via char n-grams.

#### 3.1 CRNN spec (the concrete form of A.3)

Conv front-end as a learned n-gram extractor + downsampler, then a bidirectional
GRU for full-context, order-sensitive features, then the existing two heads.

```
Embedding(38 → 32, padding_idx=0)
Conv1d(32 → 64, k=3, pad=1, stride=2) + ReLU      # 64 → 32 steps
Conv1d(64 → 64, k=3, pad=1, stride=2) + ReLU      # 32 → 16 steps
BiGRU(64 → 64, num_layers=1, batch_first=True)    # 16 steps, 128-d output
masked mean-pool ⧺ masked max-pool over GRU outputs   → 256-d
Dropout(0.2)
├─ Linear(256 → 133)   emoji
└─ Linear(256 → 8)     feeling
```

~150–250k params; still CPU-fine and ONNX-exportable. Rationale:

- **BiGRU over BiLSTM**: fewer params, faster, no long-range dependency at this
  length. 1 layer is enough; try 2 only if it underfits.
- **Pool the GRU outputs, don't take the last hidden state.** Keep the current
  masked-max-pool pattern and add masked mean-pool alongside it (concat). Mean+
  max is more robust and resists the attractor-class collapse; taking the last
  valid state forces a gather-by-length, which is what made the *old* LSTM ONNX
  export need an `ExportWrapper`.
- **Downsample the pad mask with the conv strides** (max-pool the boolean mask,
  same kernel/stride) so the GRU pooling still masks the right steps. This is
  the one fiddly bit.
- **No `pack_padded_sequence`.** Run the GRU over the full padded sequence and
  mask afterward. onnxruntime-web supports GRU (incl. bidirectional) at opset
  18; masked pooling traces cleanly with no wrapper.

Trade-offs:

| vs. | verdict |
| --- | --- |
| Dilated CNN (A.1) | CRNN is strictly more expressive for negation/order; ~2–3× slower per epoch but still seconds on 12k rows. Dilated CNN is the minimal diff; CRNN is worth it if negation matters to the product. |
| 2-layer Transformer | At 12k samples / len 64, BiGRU usually wins — no positional encoding to tune, less regularisation fuss. |

Caveats:

- A model this size on ~12k rows / 133 classes **will** overfit without the
  rest — pair it with the Dropout(0.2) above, `WEIGHT_DECAY` 1e-4 (rec E), and
  the larger deterministic eval (rec D) so the train/val gap is visible.
- CRNN does **not** lift the label-noise / near-synonym ceiling — B (prune the
  tail) and honest macro-F1 eval still gate how far top-1 can climb.

**B. Fix the label space.**
- Drop or fold emojis with < ~80 training samples; aim for 40–60 classes each
  with ≥100 samples (CLAUDE.md's intended "80" is a reasonable ceiling).
- Define explicit synonym groups (sad-faces, angry-faces, laughing-faces) and
  also score/report top-1 *within group*; 😔 vs 😞 is not recoverable from text.
- Reconcile `labels.json` (133) and CLAUDE.md (80 / LSTM / `main.py`).

**C. Grow and rebalance data.**
- Use `add-samples` / `bun run gen_data.ts`, but target the **tail**: generate
  for under-200 emojis and for Love / Neutral specifically, not random pairs.
- `WeightedRandomSampler` by inverse emoji frequency in `train_data_loader`,
  and/or inverse-freq `weight=` on both `CrossEntropyLoss` heads.
- De-duplicate near-identical texts before `split()`; make the split
  deterministic per row (hash of normalized text) so it's stable as the corpus
  grows.

**D. Make eval honest, and select on it.**
- Raise `TEST_LEN` to ~1500–2000 (10–15%).
- Report per-class emoji accuracy and **macro-F1**, not just micro top-1.
- Select the checkpoint on `feeling_acc + emoji_top3` or macro-F1, and fold the
  `emoji_keywords` battery into the selection metric so training optimises the
  thing the report measures.

### Medium impact

**E. Loss / regularisation.**
- Inverse-frequency class weights or focal loss on the emoji head.
- Weight the two task losses (e.g. `0.5·emoji + 1.0·feeling`, or uncertainty
  weighting). The 133-way emoji loss currently dominates the shared trunk.
- `WEIGHT_DECAY` → ~1e-4, add dropout 0.1–0.2 after the conv stack once
  capacity grows.
- Add `label_smoothing` to the feeling head too.

**F. Capacity.** `EMBED_SIZE` 16→32, `H_SIZE` 32→64 (or 128). Still <1M params;
the headroom is entirely in the emoji head and trunk, not the feeling head.

**G. Negation coverage** (only after A). Add training texts with negations
("not happy", "wasn't excited", "no longer sad") mapped to sensible labels;
keep the negation battery as a release gate.

### Low impact / hygiene

- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt`).
- LR schedule: cosine decay or `ReduceLROnPlateau`.
- Evaluate every epoch (not every 10) once the eval set is larger; consider an
  EMA of weights for the shipped checkpoint.
- Seed the DataLoader workers / set deterministic algorithms for comparable
  runs.

## 4. Suggested order of work

1. **D** (bigger deterministic eval + macro-F1) — you can't tell if anything
   else helps until the metric is trustworthy.
2. **B** (prune the emoji tail) — removes unlearnable noise for free.
3. **A.1/A.2** (dilated convs + mean+max pool) — one small `model.py` change,
   should move negation and multi-word cues immediately.
4. **C** (rebalance sampler + tail data) and **E** (class weights).
5. **F** (capacity), then **A.3 / §3.1** (conv + bi-GRU CRNN) if negation or
   top-3 is still short of target. Skip straight to it instead of A.1/A.2 if
   negation is a hard product requirement.
