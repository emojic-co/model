# Model improvement notes

Based on the latest behavioral report (`report/08-29-06:23.md`), the latest
tensorboard run
(`TIME: 2026-08-29 06:19:18 | DATA: mtl 42 tl 500 | MODEL: cs1 36 cs2 16 ee 64 tm 1.0 | TRAIN: lr 0.2 bs 128 gc 1.0 wd 1e-05`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json`,
`test_model.py` and `emoji_keywords.py`. The report, the shipped `model.pt`, and
that tb run are the same training (`ee 64`, finished ~06:23); the prior run in
`runs/` is `ee 32` on a smaller corpus and ran 300 epochs, so it is only a rough
point of comparison, not a clean A/B.

**What changed under the model since the previous notes.** The emoji head is
**back**, as a metric-learning head, not a second softmax:

- `F.one_hot(x, 38)[:, :, 1:]` (37 raw channels) → `Conv1d(37→36, k=3, no bias)`
  → ReLU → `MaxPool1d(k=3, stride=2)` → 1-layer unidirectional `LSTM(36→16)`,
  last state `h[-1]`.
- Feeling head: `Linear(16→8)`, plain `CrossEntropyLoss`.
- Emoji head: `emoji_proj = Linear(16→64)` gives an anchor `q`; a learned table
  `emoji_embed = nn.Embedding(300, 64)` holds one vector per emoji. Trained with
  `nn.TripletMarginLoss(margin=1.0)` where the **negative is a random other row
  of `emoji_embed`** (`emoji_embed[(target + rand 1..N-1) % N]`), not another
  example. Prediction = nearest `emoji_embed` row to `q` by L2 (`torch.cdist` in
  eval, expanded `-‖q-e‖²` in `ExportWrapper` for ONNX).
- Total training loss is the **unweighted sum** `loss_feeling + loss_emoji`.
- `CHANNELS` 32 → **36** (`HIDDEN` still 16); new `EMOJI_EMBED_SIZE = 64`,
  `TRIPLET_MARGIN = 1.0`. Params ≈ **27.9k**, of which the 300×64 emoji table is
  **19.2k** (69%).
- Optimizer switched **Adam → SGD**, `LR` 0.01 → **0.2**, flat, `EPOCHS = 100`,
  `EVAL_EPOCHS = 2`.
- Logging changed: `train.py` now logs only `train/f_loss`, `train/e_loss`,
  `eval/f_acc`, `eval/e_acc`. **`eval/f_loss`, `eval/e_loss` and `train/f_acc`
  are all commented out** — the train/eval *loss* gap and train accuracy can no
  longer be read from tb.
- `ExportBest` now selects `model.pt` on **`eval/f_acc`** (max), not eval loss
  (its own docstring and the `train.py` module docstring still say "eval feeling
  loss" — stale).
- `ExportWrapper` collapses the emoji head to an `emoji_logits` tensor and the
  ONNX contract is back to `(feeling_logits, emoji_logits)`, so **the web app no
  longer throws** — the previous notes' §2.8 is fixed.
- `labels.json`: feelings still 8; emojis now **300** (was 134).
- Corpus: `data.jsonl` **39,309** rows; after `data.py:read()` filtering
  **34,479** trainable rows (drops: 4,814 for an out-of-palette emoji, 12 for an
  out-of-set feeling, 2 both, 2 over-length). `read()` still gates on emoji —
  now legitimately, since there is an emoji head.

**Headline shift in the analysis.** Two separate stories now. **Feelings**: SGD
at `lr 0.2` removed the Adam overfitting-drift the previous notes were built
around — `eval/f_acc` climbs the whole run and peaks at the end (0.520 @ epoch
~96), `train/f_loss` plateaus at 1.29 (ln 8 = 2.08), so feelings are mildly
*under*-fit / data-or-capacity-bound, not over-fit. Also, class balance is fixed:
**Love went from 180 trainable rows to 3,392**, so the previous "Love is
unlearnable" collapse is gone (the report predicts `love`→Love now).
**Emoji**: the new head barely works — `eval/e_acc` peaks at **0.060** over 300
classes and the cue battery is **1/20**. The training signal is degenerate (see
§2.1). This is now the model's worst area by a wide margin.

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | **7/8 (88%)** | ✅ `happy`, `excited`, `calm`, `sad`, `anxious`, `neutral`, `love`. ❌ `angry`→Happy. The prior report (`01:44`, `ee 32`) was 6/8 with `love`→Excited; this 8-point check is noisy, treat 6–7/8 as the band. |
| Negations (`not <feeling>`) | **3/8** avoided the named feeling; **2/6** hit the expected opposite | Avoided **and** opposite: `not happy`→Sad, `not calm`→Anxious. Avoided (no opposite defined): `not love`→Angry. Negation is a no-op for the other 5: `not excited`→Excited, `not sad`→Sad, `not angry`→Angry, `not anxious`→Anxious, `not neutral`→Neutral. Prior report was 1/8 avoided — noise around a low mean. |
| Eval feeling accuracy (tb) | **0.504** at epoch 100; peak **0.520** @ epoch ~96 (`ExportBest` ships this checkpoint) | vs a **0.167** majority-class baseline (always-Neutral, 5,768 / 34,479 filtered rows). Real signal; rises monotonically from ~0.17, plateaus ~0.48–0.52 over the back half. No post-optimum collapse. |
| Eval emoji accuracy (tb) | peak **0.060** @ epoch ~74, **0.056** at epoch 100 | Nearest-embedding retrieval over 300 classes (chance 0.0033). ~17× chance, but useless in absolute terms. |
| Emoji cue battery (report) | top-1 **1/20**, top-3 **3/20**, top-5 **5/20** | Only `nervous`→😰 lands top-1. Negative-emotion cues keep the target in top-5 (crying 😢@3, heartbroken 💔@2, furious 😡@4, sleepy 😴@5); positive/neutral cues collapse to generic attractors — 🌮 alone is top-1 for 8 of 20 (fire, birthday, rocket, rain, meditate, down…), plus 🙂/😊/😎. |
| Train losses (tb) | `f_loss` 2.07 → **1.29** (flat by epoch ~98); `e_loss` 0.93 → **0.49** (triplet, margin 1.0) | `train/f_acc` and `eval/*_loss` are no longer logged, so the train/eval gap is not directly observable. `f_loss` well above 0 ⇒ feelings are not memorised. `e_loss` ≈ half the margin ⇒ the emoji table trivially separates without learning a useful text→emoji map. |

## 2. Root causes

### 2.1 The emoji head's training signal is degenerate (biggest, new)
`training_step` builds the triplet negative as `emoji_embed[(target_emoji +
offset) % N]` — a **random other row of the same learned table**. There is no
contrast against other examples in the batch, no hard-negative mining, and the
"negative" is a parameter the optimizer also controls. So `e_loss` falls to ~0.49
just by the 300 table rows drifting apart, while `q` (a 64-d projection of a 16-d
LSTM state) never has to resolve which emoji a given *text* means. Result:
`eval/e_acc` 0.060, battery 1/20. This is the #1 thing to fix, and it is a loss
design problem, not a capacity one.

### 2.2 One 16-d hidden state feeds both heads
`HIDDEN = 16`, unidirectional, last-step only. The same 16 numbers must carry an
8-way feeling decision **and** a 300-way emoji retrieval. Feelings plateau at
~0.50–0.52; a 300-way signal cannot fit through 16 d at all. The `mean/max`
pooling alternative is already stubbed out at `model.py:54`
(`# out = torch.max(out, dim=1).values`).

### 2.3 Unweighted sum of two differently-scaled losses
`return loss_feeling + loss_emoji` — CE (~1.3) plus triplet (~0.5), no weight, no
normalization. The feeling CE dominates the gradient; the emoji head gets the
leftover capacity of an already-tiny trunk. No `label_smoothing` on the feeling
CE either.

### 2.4 `EMOJI_EMBED_SIZE` 32 → 64 is untested and probably didn't help
The only other run in `runs/` (`ee 32`, `01:34`) reached `eval/e_acc` peak 0.118
and `eval/f_acc` 0.540, vs 0.060 / 0.520 here — but it trained **300 epochs on a
smaller corpus** (~178 steps/epoch vs 265), so this is not an A/B. At a matched
step count (~26.5k) it was at 0.098 / 0.512. What's solid: no run has pushed
emoji retrieval above ~0.12, and 64 d is 19.2k of 27.9k params spent on a head
that isn't learning. Revert to 32 (or fold into §2.1/§2.2) until the head works.

### 2.5 Feelings are plateaued ~0.50, and the previous "overfitting drift" is gone
SGD at `lr 0.2` (flat, no schedule) over 100 epochs: `eval/f_acc` rises from
~0.17 and peaks at epoch ~96; `train/f_loss` flattens at 1.29. No late eval
degradation — the Adam-era pattern (eval loss bottoming at epoch ~16 then
climbing) does not occur here. With `train/f_acc` and `eval/f_loss` no longer
logged the gap can't be measured, but every visible signal says feelings are
**under-fit / data-bound**, not over-fit. `EPOCHS = 100` is no longer mostly
wasted.

### 2.6 Class imbalance: fixed for feelings, still open for emoji
After `read()`: feelings span Neutral 5,768 … Sad 3,802 … **Love 3,392** (1.7×
head-to-tail) — Love was 180 in the previous notes. `love`→Love in the report
confirms it. Emoji: 300 classes, 56–800 rows, median 74, IQR 66–105 — far flatter
than before but still a 14× head-to-tail ratio with no per-class weighting on
either head.

### 2.7 Eval split is 500 random rows, non-stationary, no dedup
`TEST_LEN = 500` is 1.4% of 34,479. `data.py:split()` does
`random.Random(42).shuffle(read())` then takes the first 500; `data.jsonl` is
append-only (≈ +15k rows since the previous notes, and `add-samples` keeps
adding), so **every data add reshuffles the holdout** and runs across an add are
not comparable. `eval/f_acc` jitters ±0.02 between adjacent evals. No
de-duplication before the split, so near-identical generated texts can straddle
it.

### 2.8 Negation is unsupervised and OOD
`test_model.py` feeds bare `not <feeling>` (1–2 tokens) while training texts
average ~33 chars of informal sentence. Nothing in the loss rewards `not`;
`NEGATION_EXPECTED` covers 6 feelings. 3/8 avoided this run vs 1/8 last run is
noise. Keep it as a release gate, not a metric that moves on its own.

### 2.9 Minor / hygiene
- `train.py` module docstring and `ExportBest`'s docstring say the checkpoint is
  chosen "by eval feeling loss"; the code selects on `eval/f_acc` and eval loss
  is not logged. Fix the comments.
- `eval/f_loss`, `eval/e_loss`, `train/f_acc` are commented out in `train.py`.
  Re-enable at least `eval/f_loss` + `train/f_acc` — one line each, and the
  train/eval gap is the thing you most need to see.
- `data.jsonl` carries 14 out-of-set feeling rows (Annoyed 5, Confused 4,
  Frustrated 2, Hopeful/Amused/Relieved 1). `read()` drops them silently; an
  `annotation.ts` guard would stop the drift.
- `emoji_keywords.py` (~50 emoji → keyword lists) is still dead code — nothing
  imports it; `test_model.py` uses its own inline 20-pair `EMOJI_CUES`. Delete
  it, or wire its list into the battery now that there is an emoji head again.
- `CHARS` still has no digits; `VOCAB_SIZE = 38`.
- `config.py`: `EMOJI_EMBED_SIZE` / `TRIPLET_MARGIN` are live and feed
  `CONFIG_NAME` (`ee` / `tm`); the old dead `ns` field is gone — that hygiene
  item is done.
- `CLAUDE.md` is stale on nearly every technical point: it describes a learned
  `nn.Embedding` + `pack_padded_sequence` + two cross-entropy heads, `main.py` as
  the entry point, an "80-emoji palette", `gen_labels.ts` as "top 200", and
  `config.py` names `EMBED_SIZE` / `H_SIZE` / `NUM_LAYERS`. Actual: one-hot char
  → `Conv1d` → 1-layer LSTM, feeling CE + emoji **triplet**, `train.py`
  (Lightning) entry point, **300** emoji, `config.py` names `CHANNELS` /
  `HIDDEN` / `KERNEL_1` / `EMOJI_EMBED_SIZE` / `TRIPLET_MARGIN`. The
  `ExportWrapper` description (replaces `pack_padded_sequence` with a gather) is
  also wrong — it now collapses the emoji embedding head to `emoji_logits`.
- `normalize`'s run-collapse (`re.sub(r'(.)\1{2,}', r'\1\1', text)`) must stay
  byte-identical with `docs/app.js` on any future change.

## 3. Recommendations (prioritised)

### Done since the previous notes
- **Emoji head restored** (metric-learning / nearest-embedding) and
  `ExportWrapper` emits `emoji_logits` again → the deployed web app no longer
  throws on first inference (old §2.8 / rec E resolved).
- **Feeling class balance fixed** — corpus grew to 34,479 trainable rows, Love
  180 → 3,392. Old §2.4 feeling-tail collapse and the "grow the tail" part of old
  rec F are done.
- **Adam → SGD, `lr` 0.01 → 0.2** killed the post-optimum eval-loss drift the old
  notes centered on (old §2.1 / §2.2). `EPOCHS = 100` is no longer wasted.
- `read()`'s emoji gate is now correct (there is an emoji head) — old rec B
  ("stop gating `read()` on emoji") is **void**; do not do it.
- `CONFIG_NAME` hygiene: dead `ns` field removed.

### High impact

**A. Fix the emoji training signal (§2.1, §2.3) — now the #1 change.**
- Replace the table-sampled triplet negative with **in-batch negatives**: an
  InfoNCE / cross-entropy over `q @ emoji_embed.T` (temperature-scaled) gives
  every row N−1 real contrasts per step and is the standard text→label retrieval
  loss. Or, simplest and probably a higher ceiling than the current head: drop
  `emoji_proj` / `emoji_embed` for a plain `Linear(HIDDEN→300)` CE head —
  300-way CE is well-trodden and exports trivially.
- Whatever the head, **weight the two losses** (`loss_feeling + λ·loss_emoji`, or
  normalize their scales) so the emoji gradient isn't swamped by the feeling CE.

**B. Widen and enrich the shared trunk (§2.2, §2.4).**
- `HIDDEN` 16 → 48–64; make the LSTM **bidirectional**; pool `mean ⧺ max` over
  the LSTM outputs instead of `h[-1]` (stub already at `model.py:54`). Both heads
  currently read 16 numbers.
- Add a learned char embedding — `nn.Embedding(38, 24, padding_idx=0)` in place
  of `F.one_hot(x, 38)[:, :, 1:]`. Nearly free, gives the conv a real basis.
- Revert `EMOJI_EMBED_SIZE` 64 → 32 (§2.4) unless A makes it moot.

**C. Make eval honest (§2.7).**
- Deterministic per-row split: hash of the normalized text, not
  `random.Random(42).shuffle`. Stable as `data.jsonl` grows.
- De-duplicate on the normalize key before splitting.
- `TEST_LEN` 500 → ~3,000–4,000 (~10%).
- Re-enable `eval/f_loss` + `train/f_acc` logging; add **macro-F1** over the 8
  feelings, and log emoji **top-5** retrieval accuracy (top-1 0.06 hides that the
  negative-emotion cues are landing in the right neighborhood).

### Medium impact

**D. Regularize — but only after B.** Right now feelings underfit
(`train/f_loss` plateau 1.29), so dropout / heavier weight decay would hurt. Once
B gives a 48–64-d BiLSTM: `WEIGHT_DECAY` → 1e-4, `nn.Dropout(0.1–0.2)` after the
conv and on the LSTM output, `label_smoothing = 0.05` on the feeling CE.

**E. Per-class weighting for the emoji head (§2.6).** 300 classes, 14×
head-to-tail. Inverse-frequency `weight=` if A switches the emoji head to CE, or
class-balanced sampling otherwise.

**F. Targeted negation + tail data (§2.8, §2.9).** Explicit negation texts
("not happy", "wasn't excited", "no longer sad") mapped to sensible feelings;
keep the battery as a release gate. Add an `annotation.ts` guard so out-of-set
feelings stop accumulating.

### Low impact / hygiene

- Reconcile `CLAUDE.md` with the actual code (entry point `train.py`, one-hot
  char-CNN → 1-layer LSTM, feeling CE + emoji triplet, `config.py` names, 300
  emoji, real `ExportWrapper` role).
- Fix the "eval feeling loss" comments in `train.py` (selection is on
  `eval/f_acc`).
- Delete `emoji_keywords.py`, or wire its ~50-emoji list into `test_model.py`.
- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt` and
  reshuffles the split).
- `pl.seed_everything(0, workers=True)` is set, but `num_workers=4` without
  per-worker seeding still leaves run-to-run noise.

## 4. Suggested order of work

1. **C** — deterministic + larger eval, re-enable the loss / train-acc scalars,
   macro-F1, emoji top-5. Nothing below is measurable until the eval set is
   stable and the scalars are back.
2. **A** — fix the emoji loss (in-batch negatives / InfoNCE, or a plain 300-way
   CE head) + loss weighting. This is where the model is worst: `eval/e_acc`
   0.060, battery 1/20.
3. **B** — widen + bidirectionalize the trunk, learned char embedding, revert
   `ee` to 32. Unblocks both heads from the 16-d bottleneck.
4. **D** — regularize, once B makes the model large enough to overfit.
5. **E** — emoji class weighting.
6. **F** — negation + tail data, once the heads can use it.
- Hygiene (`CLAUDE.md`, stale `train.py` comments, `emoji_keywords.py`) any time.
