# Model improvement notes

Based on the latest behavioral report (`report/08-28-19:48.md`), the latest
tensorboard run
(`DATA: mtl 42 tl 500 | MODEL: cs1 16 cs2 12 ee 8 ns 16 | TRAIN: lr 0.01 bs 128 gc 1.0 wd 1e-05 | TIME: 2026-08-28 19:47:17`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json`,
`emoji_keywords.py` and `test_model.py`.

**What changed under the model since the previous notes.** The architecture in
the old `model.md` (2-layer char-CNN with a learned embedding, a contrastive
8-d emoji head, sampled softmax) is gone. The current model is a **feeling-only
classifier**: `F.one_hot(x, 38)[:, :, 1:]` → `Conv1d(37→16, k=3, no bias)` →
ReLU → `MaxPool1d(k=3, stride=2)` → 1-layer unidirectional `LSTM(16→12)` →
`Linear(12→8)`. **There is no emoji head.** ~3.3k parameters total (conv 1,776 +
LSTM 1,440 + head 104). Entry point is `train.py` (PyTorch Lightning); the
`ExportBest` callback writes `model.pt` + `docs/` whenever `eval/f_loss`
improves. `emoji_keywords.py` still exists but `test_model.py` no longer imports
it — the emoji-keyword battery is gone; the suite is now Feelings (8) +
Negations (8, 6 with an expected opposite).

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | **4/8 (50%)** | ✅ `happy`, `excited`, `anxious`, `neutral`. ❌ `calm`→Neutral, `sad`→Neutral, `love`→Neutral, `angry`→Happy. Misses collapse onto the majority classes (Neutral/Happy). |
| Negations (`not <feeling>`) | **4/8** avoided the named feeling; **0/6** hit the expected opposite | `not happy`→Happy, `not excited`→Excited, `not angry`→Angry, `not anxious`→Anxious: for half the set negation is a no-op. The 4 that "avoid" land on an unrelated feeling (`not calm`→Excited, `not sad`→Angry), never the opposite. No representation of negation. |
| Eval feeling accuracy (tb) | **0.59** at the shipped checkpoint (epoch ~39); **0.60** peak (epoch ~44); **0.578** at epoch 99 | vs a 0.197 majority-class baseline (always-Anxious on the filtered corpus). Real signal, but a hard ceiling. |
| Eval feeling loss (tb) | min **1.2222** @ step 3839 (epoch ~39), drifts up to **1.3389** by epoch 99 | Overfitting sets in after ~epoch 40; the last ~60 epochs are wasted. `ExportBest` does ship the epoch-39 checkpoint (selection is on `eval/f_loss`), so the drift doesn't reach `model.pt`. |
| Train (tb) | `f_loss` 1.93 → 0.88 (min 0.878); `f_acc` 0.22 → **0.687** (max 0.690) | The model cannot fit the training set past ~69% — a capacity floor, not just overfitting. |

Recent reports (`19:29`–`19:48`, same session) bounce between feelings 3–5/8
and negations 3–6/8 avoided / 0–2/6 opposite with no trend. This is a weak,
high-variance baseline, not a regression from a known-good state.

## 2. Root causes

### 2.1 Capacity floor — the model can't fit its own training data (biggest)
`CHANNELS = 16`, `HIDDEN = 12`, a single unidirectional LSTM layer, one-hot
input, ~3.3k parameters. `train/f_acc` plateaus at 0.687 and `train/f_loss` at
0.88 — the trunk is too small to represent 8-way sentiment over 42-char informal
text, so both train and eval are capped. Everything downstream (name prompts,
negation) is gated on this.

### 2.2 One-hot input, no learned character embedding
`model.py` feeds `F.one_hot(x, VOCAB_SIZE)[:, :, 1:]` (37 raw channels) straight
into the conv. There is no `nn.Embedding`, so the 16 conv filters must learn all
character-similarity structure from scratch. A small learned embedding
(`nn.Embedding(VOCAB_SIZE, 16..32, padding_idx=0)` in place of the one-hot) is
nearly free and gives the conv a better basis.

### 2.3 Overfitting, with no regularisation and no schedule
Same tb run: `eval/f_loss` bottoms at 1.222 (epoch ~39) then climbs to 1.34 by
epoch 99 while `train/f_loss` keeps falling to 0.88. `WEIGHT_DECAY = 1e-5` is
effectively off; there is no dropout, no `label_smoothing`, and `LR = 0.01` runs
flat for `EPOCHS = 100` with no schedule. `EVAL_EPOCHS = 5` over 100 epochs is a
lot of compute spent well past the optimum.

### 2.4 Class imbalance, unweighted loss → collapse onto the majority classes
After `data.py:read()` filtering the trainable feeling counts are Anxious 2786,
Happy 2594, Sad 2051, Angry 1916, Neutral 1678, Calm 1459, Excited 1458,
**Love 180** (15.5× under Anxious). `train.py` uses a bare
`nn.CrossEntropyLoss()` — no `weight=`, no `label_smoothing` — and
`train_data_loader` has no `WeightedRandomSampler`. The report symptoms are
exactly this: `love`, `calm`, `sad` name prompts all resolve to Neutral, and
Love (180 rows) is essentially unlearnable as is.

### 2.5 `read()` discards ~11% of usable rows on a dead constraint
`data.py:read()` keeps a row only if `d["emoji"] in emoji2idx`, but the model
has no emoji head. `data.jsonl` has 520 distinct emoji; `labels.json` lists 133;
**1,722 rows (10.9% of the rows with a valid feeling) are dropped purely because
their emoji is not one of the 133.** That is free training data thrown away, and
the drop is not uniform across feelings, so it also distorts the class balance
in 2.4. One-line fix: gate `read()` on feeling only while there is no emoji
head.

### 2.6 Eval split is 500 random rows and non-stationary
`TEST_LEN = 500` is ~3.5% of the 14,122 filtered rows. `data.py:split()` does
`random.Random(42).shuffle(data)` then takes the first 500, but `data.jsonl` is
append-only and `read()` returns rows in file order, so every `add-samples` run
reshuffles which rows are held out — runs from before and after a data add are
not comparable. `eval/f_acc` visibly jitters ±0.02–0.03 between adjacent evals.
There is no de-duplication before the split, so near-identical generated texts
can straddle it (mild leak).

### 2.7 The name / negation batteries are out-of-distribution and unsupervised
`test_model.py` feeds bare tokens (`"calm"`, `"not happy"`) — 1–2 words, no
punctuation — while training texts average ~33 chars of informal sentence.
Nothing in the loss rewards handling `not`, and there are few explicit
negation constructions in `data.jsonl`. 0/6 opposites is the expected result
until the corpus contains negation examples; treat the negation battery as a
release gate, not a metric that will move on its own.

### 2.8 The deployed web app is broken against the current export
`train.py:export_onnx` emits a **single** output, `feeling_logits`. But
`docs/app.js:87` reads `out.emoji_logits.data` and `renderDebug` (`app.js:105`,
`:112`) renders an emoji probability row from it. `out.emoji_logits` is now
`undefined`, so `argmax(undefined.data)` throws on the first inference. The
emoji `<div>`, `meta.json`'s `emojis` list, and the debug emoji panel are all
dead weight. Either restore an emoji head or strip the emoji path from
`app.js` / `index.html` / `export_web`.

### 2.9 Minor / hygiene
- `CHARS` in `data.py` has no digits (`0-9` are dropped by `normalize`);
  `VOCAB_SIZE = 38`.
- `emoji_keywords.py` is dead code — `test_model.py` no longer imports it.
  Delete it, or wire the battery back in once an emoji head returns.
- `CLAUDE.md` is stale on nearly every technical point: it describes an LSTM
  with a learned embedding + `pack_padded_sequence` + `ExportWrapper`, two heads
  (emoji + feeling), `main.py` as the entry point, an "80-emoji palette", and
  `config.py` names (`EMBED_SIZE` / `H_SIZE` / `NUM_LAYERS`). None of that
  matches the code. `labels.json` has 133 emoji while `gen_labels.ts` is
  documented as "top 100".
- `config.py`'s `EMOJI_EMBED_SIZE` / `NEGATIVE_SAMPLES` are marked "NOT IN USE"
  but still feed `CONFIG_NAME` (`ee 8 ns 16` in every run name).
- `normalize`'s run-collapse (`re.sub(r'(.)\1{2,}', r'\1\1', text)`) must stay
  byte-identical with `docs/app.js` on any future change.

## 3. Recommendations (prioritised)

### Done since the previous notes
- Emoji head removed entirely; training is feeling-only (side-steps the old
  §2.1/§2.2 contrastive-bottleneck problems — at the cost of shipping a broken
  emoji path in the web app, see §2.8).
- Checkpoint selection is on `eval/f_loss` (a single, informative term now that
  the noisy emoji-loss term is gone), and `ExportBest` correctly ships the
  epoch-39 optimum.
- `MAX_TEXT_LEN` 64 → 42 (matches the data; `scan_text_len.py` covers this).

### High impact

**A. Raise trunk capacity — this is the ceiling.**
1. Learned embedding instead of one-hot: `nn.Embedding(VOCAB_SIZE, 24,
   padding_idx=0)` → conv. Nearly free, addresses §2.2.
2. `CHANNELS` 16 → 48–64, `HIDDEN` 12 → 48–64, and make the LSTM
   **bidirectional** (pool `mean ⧺ max` over its outputs rather than taking the
   last state — keeps the ONNX export a plain trace and lets negation be
   represented). Still well under 1M params.
3. Re-check `train/f_acc`: if it climbs past ~0.8 the capacity floor is lifted;
   if not, the data (§2.4/§2.5) is the limit.

**B. Fix the loss and the schedule (§2.3, §2.4).**
- Inverse-frequency `weight=` on `CrossEntropyLoss` (or `WeightedRandomSampler`
  in `train_data_loader`), plus `label_smoothing=0.05–0.1`. Directly targets the
  `love`/`calm`/`sad` → Neutral collapse.
- `WEIGHT_DECAY` → 1e-4; add `nn.Dropout(0.1–0.2)` after the conv/LSTM once
  capacity grows.
- Cosine or `ReduceLROnPlateau` LR schedule; or just cut `EPOCHS` to ~40 — the
  optimum is at epoch ~39 and everything after is waste.

**C. Stop discarding data (§2.5).** Gate `read()` on `feeling in feeling2idx`
only. Recovers ~1,722 rows (+12%) and de-skews the class balance for free.

**D. Make eval honest (§2.6).**
- Deterministic per-row split: hash of the normalised text, not
  `random.Random(42).shuffle`. Stable as `data.jsonl` grows.
- De-duplicate on the normalise key before splitting.
- Raise `TEST_LEN` to ~1500–2000 (~10–15%).
- Log macro-F1 across the 8 feelings alongside accuracy — with Love at 180 rows,
  micro-accuracy hides the tail.

### Medium impact

**E. Fix or remove the web-app emoji path (§2.8).** Until an emoji head is back,
strip `emoji_logits` handling from `app.js`, drop the emoji `<div>` and debug
panel, and stop emitting `emojis` from `export_web` / `meta.json`. The Pages
deploy currently ships a page that errors on first use.

**F. Grow the corpus toward the tail (`add-samples` / `bun run gen_data.ts`).**
Target Love, Calm, Excited, Neutral specifically, not random pairs. Also add
explicit negation texts ("not happy", "wasn't excited", "no longer sad") mapped
to sensible feelings, and keep the negation battery as a release gate (§2.7).

### Low impact / hygiene

- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt` and
  reshuffles the split).
- Reconcile `CLAUDE.md` with the actual code (entry point, architecture, head
  count, `config.py` names, emoji count) and delete `emoji_keywords.py` or
  re-wire it.
- Drop the unused `ee`/`ns` fields from `CONFIG_NAME`.
- Seed DataLoader workers / set deterministic algorithms for fully comparable
  runs (`pl.seed_everything(0, workers=True)` is already set).

## 4. Suggested order of work

1. **C** — one-line `read()` fix. Free data, no downside, do it first.
2. **D** — deterministic bigger eval + macro-F1. Nothing else is measurable
   until the eval set is stable and large enough to see the Love/Calm tail.
3. **A** — embedding + wider bidirectional trunk. This is the change that lifts
   the 0.60 eval-accuracy / 0.69 train-accuracy ceiling and makes negation
   representable.
4. **B** — class weights + `label_smoothing` + shorter schedule / LR decay.
   Kills the majority-class collapse and stops the epoch-40+ drift.
5. **F** — targeted tail + negation data, once the model can actually use it.
6. **E** — repair the web app (independent of the modelling work; can be done
   any time, and should be soon since Pages is serving a broken page).
