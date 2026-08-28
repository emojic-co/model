# Model improvement notes

Based on the latest behavioral report (`report/08-28-20:53.md`), the latest
tensorboard run
(`TIME: 2026-08-28 20:51:48 | DATA: mtl 42 tl 500 | MODEL: cs1 32 cs2 16 ee 8 ns 16 | TRAIN: lr 0.01 bs 128 gc 1.0 wd 1e-05`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json`,
`emoji_keywords.py` and `test_model.py`.

**What changed under the model since the previous notes.** Still a
**feeling-only classifier**, no emoji head:
`F.one_hot(x, 38)[:, :, 1:]` (37 raw channels) → `Conv1d(37→32, k=3, no bias)` →
ReLU → `MaxPool1d(k=3, stride=2)` → 1-layer unidirectional `LSTM(32→16)` →
`Linear(16→8)`. `CHANNELS` went 16 → **32** and `HIDDEN` 12 → **16**
(`config.py`), so params roughly doubled: ~6.9k (conv 3,552 + LSTM 3,200 + head
136). `EVAL_EPOCHS` went 5 → **2**. Entry point is still `train.py` (PyTorch
Lightning); the `ExportBest` callback writes `model.pt` + `docs/` whenever
`eval/f_loss` improves. `test_model.py` is unchanged: Feelings (8) + Negations
(8, 6 with an expected opposite); `emoji_keywords.py` is still not imported.

**Headline shift in the analysis.** The capacity bump lifted the *training*
ceiling (`train/f_acc` 0.69 → 0.78, `train/f_loss` 0.88 → 0.64) but not
generalisation (`eval/f_acc` peak 0.60 → 0.602, `eval/f_loss` best 1.222 →
1.170). The extra width was spent memorising the train set, and the
post-optimum drift got **worse and earlier** (eval loss now bottoms at epoch
~16 and climbs to 1.51 by epoch 100, vs epoch ~39 → 1.34 before). The model is
now **regularisation- and data-bound, not width-bound**.

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | **6/8 (75%)** | ✅ `happy`, `excited`, `calm`, `sad`, `anxious`, `neutral`. ❌ `angry`→Sad, `love`→Neutral. Across tonight's five reports (`20:42`–`20:53`) this bounces **5–7/8** with no trend. |
| Negations (`not <feeling>`) | **4/8** avoided the named feeling; **2/6** hit the expected opposite | Avoided **and** hit the opposite: `not happy`→Sad, `not calm`→Anxious. Avoided (no opposite defined): `not neutral`→Sad, `not love`→Sad. Not avoided — negation is a no-op: `not excited`→Excited, `not sad`→Sad, `not angry`→Angry, `not anxious`→Anxious. Session range **3–5/8** avoided, **0–2/6** opposite: noise around chance. |
| Eval feeling accuracy (tb) | **0.586** at the shipped checkpoint (epoch 16); flat **0.55–0.60** thereafter (noise peak 0.602 @ epoch 86); **0.560** at epoch 100 | vs a **0.197** majority-class baseline (always-Anxious, 2,786 / 14,120 filtered rows). Real signal, hard ceiling. |
| Eval feeling loss (tb) | min **1.1696** @ epoch 16 (step 1695), rises ~monotonically to **1.5146** @ epoch 100 | Overfitting from epoch ~16. `ExportBest` selects on `eval/f_loss`, so `model.pt` is the epoch-16 checkpoint and the drift does not ship — but the last ~84 epochs are pure waste and actively raise eval loss. |
| Train (tb) | `f_loss` 1.93 → **0.637** (still falling at epoch 100); `f_acc` 0.23 → **0.780** (still rising) | The train/eval gap widens for the entire run. Capacity is no longer the immediate limit; generalisation is. |

## 2. Root causes

### 2.1 Overfitting dominates — the capacity bump helped train, not eval (biggest)
Since the previous notes `CHANNELS` 16 → 32 and `HIDDEN` 12 → 16 (params
~3.3k → ~6.9k). Effect on the latest tb run: `train/f_acc` ceiling 0.69 → 0.78,
`train/f_loss` 0.88 → 0.64 (both still improving at epoch 100), while
`eval/f_acc` peak is unchanged (0.60 → 0.602) and `eval/f_loss` best barely
moved (1.222 → 1.170) but its post-optimum drift is worse (to 1.51 vs 1.34) and
starts earlier (epoch ~16 vs ~39). The added capacity went into memorisation.
More raw width without regularisation will keep widening the gap.

### 2.2 No regularisation, no schedule
`WEIGHT_DECAY = 1e-5` is effectively off; there is no dropout, no
`label_smoothing`, and `LR = 0.01` runs flat for `EPOCHS = 100` with no
schedule. The eval-loss optimum is at epoch ~16 — **~84% of the run is wasted**
and pushes eval loss up. With `EVAL_EPOCHS = 2` that is 42 validation passes
spent past the optimum, a lot of CPU.

### 2.3 One-hot input, no learned character embedding
`model.py` feeds `F.one_hot(x, VOCAB_SIZE)[:, :, 1:]` (37 raw channels) straight
into `Conv1d(37→32, k=3, bias=False)`. There is no `nn.Embedding`, so the conv
filters must learn all character-similarity structure from scratch. A small
learned embedding (`nn.Embedding(VOCAB_SIZE, 24, padding_idx=0)` in place of the
one-hot) is nearly free and gives the conv a better basis. Lower priority than
§2.1/§2.2 now.

### 2.4 Class imbalance, unweighted loss → tail classes unlearned
After `data.py:read()` filtering the trainable feeling counts are Anxious 2786,
Happy 2594, Sad 2051, Angry 1916, Neutral 1676, Calm 1459, Excited 1458,
**Love 180** (15.5× under Anxious). `train.py` uses a bare
`nn.CrossEntropyLoss()` — no `weight=`, no `label_smoothing` — and
`train_data_loader` has no `WeightedRandomSampler`. `love`→Neutral in the latest
report is exactly this; Love at 180 rows is essentially unlearnable as is.
(`angry`→Sad this run is not a majority-class collapse, but the tail is still
where the misses concentrate.)

### 2.5 `read()` discards ~11% of usable rows on a dead constraint
`data.py:read()` keeps a row only if `d["emoji"] in emoji2idx`, but the model
has no emoji head. `data.jsonl` has 520 distinct emoji; `labels.json` lists 134;
**1,722 rows (10.9% of the feeling-valid, length-ok rows) are dropped purely
because their emoji is not one of the 134.** A feeling-only `read()` would keep
**15,842** rows vs **14,120** now (+1,722, +12.2%). The drop is not uniform
across feelings, so it also distorts the class balance in §2.4. One-line fix:
gate `read()` on feeling only while there is no emoji head.

### 2.6 Eval split is 500 random rows and non-stationary
`TEST_LEN = 500` is ~3.5% of the 14,120 filtered rows. `data.py:split()` does
`random.Random(42).shuffle(data)` then takes the first 500, but `data.jsonl` is
append-only and `read()` returns rows in file order, so every `add-samples` run
reshuffles which rows are held out — runs from before and after a data add are
not comparable. `eval/f_acc` jitters ±0.02 between adjacent evals. There is no
de-duplication before the split, so near-identical generated texts can straddle
it (mild leak).

### 2.7 The name / negation batteries are out-of-distribution and unsupervised
`test_model.py` feeds bare tokens (`"calm"`, `"not happy"`) — 1–2 words, no
punctuation — while training texts average ~33 chars of informal sentence.
Nothing in the loss rewards handling `not`, and there are few explicit negation
constructions in `data.jsonl`. `NEGATION_EXPECTED` defines opposites for
Happy/Sad/Calm/Anxious/Angry/Excited; 2/6 hit this run (0–2/6 across the
session) is noise. Treat the negation battery as a release gate, not a metric
that will move on its own.

### 2.8 The deployed web app is broken against the current export
`train.py:export_onnx` emits a **single** output, `feeling_logits`. But
`docs/app.js:87` reads `out.emoji_logits.data`, and `renderDebug`
(`app.js:91`, `:105`, `:112`) renders an emoji probability row from it.
`out.emoji_logits` is now `undefined`, so `argmax(undefined.data)` throws on the
first inference. `export_web` still emits `"emojis"` into `meta.json`; the emoji
`<div>` and debug panel are dead weight. Either restore an emoji head or strip
the emoji path from `app.js` / `index.html` / `export_web`.

### 2.9 Minor / hygiene
- `data.jsonl` now carries out-of-set feelings — Annoyed 5, Confused 4,
  Frustrated 1, Hopeful 1 (11 rows) — outside `labels.json`'s 8. `read()` drops
  them silently; worth an `annotation.ts` guard so they don't accumulate.
- `CHARS` in `data.py` has no digits (`0-9` are dropped by `normalize`);
  `VOCAB_SIZE = 38`.
- `emoji_keywords.py` is dead code — `test_model.py` does not import it. Delete
  it, or wire the ~50-emoji battery back in once an emoji head returns.
- `CLAUDE.md` is stale on nearly every technical point: it describes an LSTM
  with a learned embedding + `pack_padded_sequence` + `ExportWrapper`, two heads
  (emoji + feeling), `main.py` as the entry point, an "80-emoji palette", and
  `config.py` names (`EMBED_SIZE` / `H_SIZE` / `NUM_LAYERS`). Actual: char-CNN →
  1-layer LSTM, feeling head only, `train.py` (Lightning) entry point,
  `config.py` names `CHANNELS` / `HIDDEN` / `KERNEL_1`. `labels.json` has 134
  emoji while `gen_labels.ts` is documented as "top 100".
- `config.py`'s `EMOJI_EMBED_SIZE` / `NEGATIVE_SAMPLES` are marked "NOT IN USE"
  but still feed `CONFIG_NAME` (`ee 8 ns 16` in every run name).
- `normalize`'s run-collapse (`re.sub(r'(.)\1{2,}', r'\1\1', text)`) must stay
  byte-identical with `docs/app.js` on any future change.

## 3. Recommendations (prioritised)

### Done since the previous notes
- `CHANNELS` 16 → 32, `HIDDEN` 12 → 16 (part of the old rec A.2). Lifted the
  train-set ceiling (`train/f_acc` 0.69 → 0.78) but **not** eval — see §2.1. The
  rest of old rec A still stands, and regularisation (below) is now the binding
  constraint.
- `EVAL_EPOCHS` 5 → 2 (finer eval curve).
- Still standing from before: checkpoint selection on `eval/f_loss`;
  `MAX_TEXT_LEN` = 42 (matches the data).

### High impact

**A. Regularise and cut the schedule (§2.1, §2.2, §2.4) — now the #1 change.**
- Inverse-frequency `weight=` on `CrossEntropyLoss` (or `WeightedRandomSampler`
  in `train_data_loader`), plus `label_smoothing=0.05–0.1`. Directly targets the
  `love` / `calm` / `excited` tail collapse.
- `WEIGHT_DECAY` → 1e-4; add `nn.Dropout(0.1–0.3)` after the conv and on the
  LSTM output state.
- Cut `EPOCHS` to ~25–30 (the optimum is epoch ~16), or add a cosine /
  `ReduceLROnPlateau` schedule. Everything past epoch ~18 only raises eval loss.

**B. Stop discarding data (§2.5).** Gate `read()` on `feeling in feeling2idx`
only. Recovers 1,722 rows (+12.2%) and de-skews the class balance for free.
One line, no downside — do it first.

**C. Make eval honest (§2.6).**
- Deterministic per-row split: hash of the normalised text, not
  `random.Random(42).shuffle`. Stable as `data.jsonl` grows.
- De-duplicate on the normalise key before splitting.
- Raise `TEST_LEN` to ~1500–2000 (~10–15%).
- Log macro-F1 across the 8 feelings alongside accuracy — with Love at 180 rows,
  micro-accuracy hides the tail.

**D. Raise trunk quality, carefully (§2.3) — after A.**
- Learned embedding instead of one-hot: `nn.Embedding(VOCAB_SIZE, 24,
  padding_idx=0)` → conv. Nearly free.
- Make the LSTM **bidirectional** and pool `mean ⧺ max` over its outputs rather
  than taking the last state — lets negation be represented, keeps the ONNX
  export a plain trace.
- Only widen `CHANNELS` / `HIDDEN` further once A is in place; right now more
  width just feeds §2.1. Re-check the `train/f_acc` − `eval/f_acc` gap after
  each change.

### Medium impact

**E. Fix or remove the web-app emoji path (§2.8).** Until an emoji head is back,
strip `emoji_logits` handling from `app.js`, drop the emoji `<div>` and debug
panel, and stop emitting `"emojis"` from `export_web` / `meta.json`. The Pages
deploy currently ships a page that throws on first inference.

**F. Grow the corpus toward the tail (`add-samples`).** Target Love, Calm,
Excited, Neutral specifically, not random pairs. Add explicit negation texts
("not happy", "wasn't excited", "no longer sad") mapped to sensible feelings,
and keep the negation battery as a release gate (§2.7). Add an `annotation.ts`
guard so out-of-set feelings (§2.9) stop accumulating.

### Low impact / hygiene

- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt` and
  reshuffles the split).
- Reconcile `CLAUDE.md` with the actual code (entry point `train.py`, char-CNN →
  1-layer LSTM, feeling head only, `config.py` names, 134 emoji) and delete
  `emoji_keywords.py` or re-wire it.
- Drop the unused `ee` / `ns` fields from `CONFIG_NAME`.
- Seed DataLoader workers for fully comparable runs
  (`pl.seed_everything(0, workers=True)` is already set; `num_workers=4` without
  per-worker seeding).

## 4. Suggested order of work

1. **B** — one-line `read()` fix. Free data, no downside, do it first.
2. **C** — deterministic bigger eval + macro-F1. Nothing else is measurable
   until the eval set is stable and large enough to see the Love/Calm tail.
3. **A** — regularisation + class weights + short schedule / LR decay. This is
   the change that attacks the dominant failure now: the widening train/eval gap
   (§2.1/§2.2) and the tail collapse (§2.4).
4. **D** — embedding + bidirectional trunk, once A stops the gap from widening
   and makes negation representable.
5. **F** — targeted tail + negation data, once the model can actually use it.
6. **E** — repair the web app (independent of the modelling work; can be done
   any time, and should be soon since Pages is serving a broken page).
