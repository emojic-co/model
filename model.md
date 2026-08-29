# Model improvement notes

Based on the latest behavioral report (`report/08-29-19:54.md`), the latest
completed tensorboard run
(`TIME: 2026-08-29 19:47:42 | DATA: mtl 42 tl 900 | MODEL: cs (256, 32) | TRAIN: lr 0.1 bs 128 gc 1.0 wd 0.0002`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json`
and `test_model.py`. That run trained **50 epochs**, wrote
the shipped `model.pt`, and its end-of-run `test_model.run()` produced the
`19:54` report — so the report, `model.pt` and the tb run are one training.
`runs/` also holds a sibling `cs (100, 50)` run (same lr/bs, 50 epochs) and a
4-epoch `19:54:42` fragment that was aborted before its first report; neither is
a clean A/B.

**What changed under the model since the previous notes.** The LSTM trunk is
**gone**. The model is now a pure char-CNN with a global max-pool:

- `nn.Embedding(38, 16, padding_idx=0)` (a **learned** char embedding, not the
  old `F.one_hot`) → transpose → `Conv1d(16→256, k=3, stride=2, no bias)` → ReLU
  → `Conv1d(256→32, k=4, stride=2, no bias)` → ReLU → `torch.max(dim=-1)` over
  the time axis → `(B, 32, 1)`.
- Feeling head: `Conv1d(32→8, k=1)` (was `Linear`), plain `CrossEntropyLoss`.
- Emoji head: `Conv1d(32→32, k=1)` → L2-normalize = anchor `q`; a learned table
  `emoji_embed = nn.Embedding(300, 32)`, rows L2-normalized. Trained with
  `nn.TripletMarginLoss(margin=0.5)` where each row now draws
  **`EMOJI_NEGATIVES = 5`** negatives, each still a random other row of
  `emoji_embed` (`emoji_embed[(target + rand 1..N-1) % N]`), tiled to `(B*5, D)`
  and meaned. Prediction = nearest `emoji_embed` row to `q` (L2 on unit vectors
  = cosine; `torch.cdist` in eval / test, `q @ emoji_embed.T` in
  `ExportWrapper`).
- Total loss is still the **unweighted sum** `loss_feeling + loss_emoji`.
- `config.py` renamed/retuned: `KERNEL_1` → `KERNELS = (3, 4)`;
  `CHANNELS` `(128, 64)` → **`(256, 32)`**; `CHAR_EMBED_SIZE` 20 → **16**;
  `EMOJI_EMBED_SIZE` 64 → **32**; `TRIPLET_MARGIN` 1.0 → **0.5**; new
  `EMOJI_NEGATIVES = 5`; `LR` 0.2 → **0.1**; `EPOCHS` → **30** in the working
  tree (the shipped run logged 50 — reconcile). There is no `HIDDEN` any more;
  the trunk output is `CHANNELS[1] = 32` channels.
- Params ≈ **56.6k**. The two convs are **45.1k (79.6%)** — `conv2` (32×256×4)
  alone is 32.8k. The 300×32 emoji table is **9.6k (17%)**. Both heads are
  tiny (feeling 264, emoji proj 1,056).
- Optimizer stays SGD (`lr 0.1`, `wd 2e-4`, flat, `grad_clip 1.0`).
- Logging: `train/f_acc` is **back** (was commented out last notes); tb now has
  `train/{f,e}_acc`, `train/{f,e}_loss`, `eval/{f,e}_acc`. **`eval/f_loss` and
  `eval/e_loss` are still commented out** in `validation_step`.
- `ExportBest` still selects `model.pt` on **`eval/f_acc`** (max); its docstring
  and `train.py`'s still say "eval feeling loss" — stale.
- `labels.json`: feelings still 8 (now includes **Love**); emojis **300**.
- `normalize` gained a run-collapse `re.sub(r'(.)\1{2,}', r'\1\1', text)` — must
  stay byte-identical with `docs/app.js`.
- Corpus: `data.jsonl` **67,439** rows; after `data.py:read()`
  (feeling ∈ 8, emoji ∈ 300, normalized len ≤ 42) **59,888** trainable
  (~7.5k dropped, almost all for an out-of-palette emoji — 857 distinct emojis
  in the raw file, 300 kept; plus 14 out-of-set feeling rows).

**Headline.** Two stories, both worse than the last notes claimed.
**Feelings**: `eval/f_acc` climbs 0.364 → 0.483 (peak **0.497**), vs a ~0.24
always-Neutral baseline — real signal but plateaued near the trunk's ceiling,
with a widening `train`/`eval` gap (`train/f_acc` 0.656) and a slight droop off
the peak. The name battery is **6/8** (`angry`→Neutral, `love`→Neutral).
**Emoji**: the head still does not learn a text→emoji map — `eval/e_acc`
**0.051** over 300 classes (~15× chance), the triplet loss barely moves
(0.468 → 0.395, still ~0.8× the 0.5 margin), and the cue battery is **3/20**
top-1, 4/20 top-5. Raising `EMOJI_NEGATIVES` 1 → 5 did nothing because all 5
negatives are still rows of the same learned table (§2.1).

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name | **6/8 (75%)** | ✅ `happy`, `excited`, `calm`, `sad`, `anxious`, `neutral`. ❌ `angry`→Neutral, `love`→Neutral. 8-point check, noisy — the prior report was 7/8; treat 6–7/8 as the band. `love` now fails as a model miss, not a data gap (Love has 7,590 trainable rows). |
| Negations (`not <feeling>`) | **2/8** avoided the named feeling; **0/6** hit the expected opposite | Only `not neutral`→Sad and `not love`→Angry "avoid" — and only because no opposite is defined for them. All 6 `NEGATION_EXPECTED` cases return the negated feeling unchanged (`not happy`→Happy, `not sad`→Sad, …). Negation is a pure no-op; 2/8 is noise. |
| Eval feeling accuracy (tb) | **0.483** at epoch 50; peak **0.497** (`ExportBest` ships this checkpoint) | vs a **~0.24** majority-class baseline (always-Neutral, 14,482 / 59,888 filtered rows). Rises from 0.36, plateaus ~0.47–0.50 over the back half, then droops ~1 pt off the peak. |
| Eval emoji accuracy (tb) | peak & final **0.051** at epoch 50 | Nearest-embedding retrieval over 300 classes (chance 0.0033). ~15× chance, useless in absolute terms. The `cs (100, 50)` sibling run also lands at 0.051 — architecture is not the lever here. |
| Emoji cue battery (report) | top-1 **3/20**, top-3 **4/20**, top-5 **4/20** | top-1: `party`→🎉, `furious`→😡, `coffee`→☕. `rain`→🌧️ lands at rank 3; nothing else puts the target in the top 5. Positive/neutral cues collapse to hub vectors — 🤩 is top-1 for 4 of 20 cues (`sleepy`, `fire`, `whatever`, `down`); 💔, 💌, 😰 for 2 each. |
| Train losses (tb) | `f_loss` 1.951 → **0.979** (still falling at epoch 50; `ln 8` = 2.08); `e_loss` 0.468 → **0.395** (triplet, margin 0.5) | `train/f_acc` 0.264 → **0.656** vs `eval/f_acc` 0.483 ⇒ ~17-pt gap, widening. `f_loss` well above 0 ⇒ feelings are not memorised — both a mild generalization gap *and* headroom. `e_loss` ≈ 0.8× margin ⇒ the 300 table rows trivially separate without a useful text→emoji map. |

## 2. Root causes

### 2.1 The emoji triplet signal is degenerate (biggest, unchanged)
`training_step` builds every negative as
`emoji_embed[(target_emoji + offset) % N]` — a **random other row of the same
learned table**, now sampled 5× per anchor instead of 1×. There is still no
contrast against other *texts* in the batch, no hard-negative mining, and the
"negative" is a parameter the optimizer also controls. So `e_loss` falls to
~0.40 just from the 300 table rows drifting apart, while `q` (a 32-d projection
of a lossy 32-channel max-pool) never has to resolve which emoji a given *text*
means. Result: `eval/e_acc` 0.051, battery 3/20. This is a loss-design problem,
not a capacity one, and bumping `EMOJI_NEGATIVES` confirmed it — more negatives
from the same source changed nothing.

### 2.2 Global max-pool over a ~9-char window = bag-of-n-grams
`Embedding → Conv1d(k3,s2) → Conv1d(k4,s2) → max over time`. Each of the 9
surviving time steps sees ~9 input characters; the max then keeps only the
single most-activating window per channel. No recurrence (the LSTM the previous
notes described is gone), no position, no long-range composition. Negation,
word order and scope past ~9 chars are structurally invisible — hence 0/6
defined negations and the positive/neutral cue collapse.

### 2.3 The trunk is 32 channels feeding both heads
`CHANNELS[1] = 32`. The same 32 numbers (after a lossy max-pool) must carry an
8-way feeling decision **and** a 300-way emoji retrieval. Feelings plateau
~0.50; a 300-way signal cannot fit through 32 d at all. `CHANNELS` `(128, 64)` →
`(256, 32)` since the last notes *narrowed* the shared representation.

### 2.4 Unweighted sum of two differently-scaled losses
`return loss_feeling + loss_emoji` — CE (~1.0) plus triplet (~0.4), no weight,
no normalization. The feeling CE dominates the gradient; the emoji head gets the
leftover capacity of an already-tiny trunk. No `label_smoothing` on the feeling
CE either.

### 2.5 Feelings are plateaued ~0.50 with a mild train/eval gap
`train/f_acc` 0.656 vs `eval/f_acc` 0.483 (~17 pt, widening), and `eval/f_acc`
peaks around epoch ~46 then droops ~1 pt — a small overfitting onset in the back
third. But `train/f_loss` is still 0.98 (well above 0), so the model is also
signal-/capacity-bound, not memorising. Net: ~0.50 is close to this trunk's
ceiling on this label set. With `eval/f_loss` not logged (§2.8) the true optimum
epoch can't be seen.

### 2.6 Class balance: solved for feelings, mild for emoji
After `read()`, feelings span Neutral **14,482** … Excited 8,209 … Angry
**6,999** — 2.1× head-to-tail, and **Love is fully populated at 7,590** (it was
180 two notes ago). The old "Love unlearnable" collapse is gone; `love`→Neutral
is now a capacity miss. Emoji palette counts are tight: min **125**, median
**150**, quartiles 142 / 150 / 168, with a short high tail (😤 1,099, 🎉/😌
next). 0 of the 300 palette emojis have zero rows. Emoji imbalance is no longer
the story — §2.1 is.

### 2.7 Eval split is 900 random rows, non-stationary, no dedup
`TEST_LEN = 900` is 1.5% of 59,888. `data.py:split()` does
`random.Random(42).shuffle(read())` then takes the first 900; `data.jsonl` is
append-only (`add-samples` keeps growing it — +28k rows since the last notes),
so **every data add reshuffles the holdout** and runs across an add are not
comparable. `eval/f_acc` jitters ~±0.015 between adjacent evals. No
de-duplication on the `normalize` key before the split, so near-identical
generated texts can straddle it.

### 2.8 Eval losses are not logged
`validation_step` comments out `eval/f_loss` and `eval/e_loss`; only
`eval/{f,e}_acc` are logged. `train/f_acc` is back this time, so the *accuracy*
gap is visible — but the train/eval **loss** gap, the single most useful
overfitting signal and the thing `ExportBest`'s docstring claims to select on,
still can't be read.

### 2.9 Negation is unsupervised and OOD
`test_model.py` feeds bare `not <feeling>` (1–2 tokens) while training texts are
informal sentences. Nothing in the loss rewards `not`; `NEGATION_EXPECTED`
covers 6 feelings. Keep the battery as a release gate, not a metric that moves
on its own — 2/8 vs 3/8 last run is noise.

### 2.10 Minor / hygiene
- `ExportBest`'s docstring and `train.py`'s module docstring say the checkpoint
  is chosen "by eval feeling loss"; the code selects on `eval/f_acc` and eval
  loss is not logged. Fix the comments.
- The emoji cue battery is `test_model.py`'s own inline 20-pair `EMOJI_CUES`.
  The old hand-curated ~50-emoji `emoji_keywords.py` was dead code and has been
  deleted; wire a larger cue list back in here if the emoji head starts working.
- `data.jsonl` carries 14 out-of-set feeling rows (Annoyed 5, Confused 4,
  Frustrated 2, Hopeful/Amused/Relieved 1). `read()` drops them silently; an
  `annotation.ts` guard would stop the drift.
- `CHARS` still has no digits; `VOCAB_SIZE = 38`.
- `config.py` working-tree `EPOCHS = 30`, but all three runs in `runs/` logged
  50 epochs (23,000 steps at 460 steps/epoch). Reconcile.
- `num_workers=4` without per-worker seeding leaves run-to-run noise despite
  `pl.seed_everything(0, workers=True)`.
- `CLAUDE.md` is stale on nearly every technical point: it describes a
  `pack_padded_sequence` LSTM with two cross-entropy heads, `main.py` as the
  entry point, an "80-emoji palette", `gen_labels.ts` as "top 200", and
  `config.py` names `EMBED_SIZE` / `H_SIZE` / `NUM_LAYERS`. Actual: learned
  `nn.Embedding` char-CNN → two strided `Conv1d` → global max-pool (no LSTM),
  feeling CE + emoji **triplet** (5 table negatives), `train.py` (Lightning)
  entry, **300** emojis, `config.py` names `CHANNELS` / `KERNELS` /
  `CHAR_EMBED_SIZE` / `EMOJI_EMBED_SIZE` / `TRIPLET_MARGIN` / `EMOJI_NEGATIVES`.
  The `ExportWrapper` description is also wrong — it collapses the emoji
  embedding head to `emoji_logits`, it does not touch `pack_padded_sequence`.

## 3. Recommendations (prioritised)

### Done since the previous notes
- **LSTM trunk removed** — replaced by a second strided `Conv1d` + global
  max-pool. Learned `nn.Embedding` replaces `F.one_hot` (old rec B's "learned
  char embedding" item).
- **`EMOJI_EMBED_SIZE` 64 → 32** (old §2.4 / rec B revert done);
  `TRIPLET_MARGIN` 1.0 → 0.5; `LR` 0.2 → 0.1.
- **`train/f_acc` logging re-enabled** (part of old rec C).
- **Emoji metric switched to top-5** — `{train,eval}/e_acc5` (target in the 5
  nearest embeddings) replaces the top-1 `e_acc`, reusing the
  `q @ emoji_embed.T` matmul already computed, so it's free (rec C item).
- **Feeling class balance holds** — Love fully populated (7,590 rows).
- Not done and still #1: the emoji loss (old rec A). `EMOJI_NEGATIVES` 1 → 5 is
  *not* the fix — all 5 negatives are still table rows (§2.1).

### High impact

**A. Fix the emoji training signal (§2.1, §2.4) — still the #1 change.**
- Replace the table-sampled triplet negatives with **in-batch negatives**: an
  InfoNCE / cross-entropy over `q @ emoji_embed.T` (temperature-scaled) gives
  every row `B−1` real *text* contrasts per step and is the standard text→label
  retrieval loss. Or — simplest, and probably a higher ceiling than the current
  head — drop `emoji` proj + `emoji_embed` for a plain `Conv1d(32→300, k=1)` CE
  head. 300-way CE is well-trodden and `ExportWrapper` collapses to a no-op.
- Whatever the head, **weight the two losses** (`loss_feeling + λ·loss_emoji`,
  or normalize their scales) so the emoji gradient isn't swamped by the feeling
  CE.

**B. Give the shared representation room (§2.2, §2.3).**
- Widen `CHANNELS[1]` 32 → 64–128; both heads currently read 32 numbers after a
  lossy max-pool, and one of them is 300-way.
- Pool `mean ⧺ max` over the conv time axis instead of `max` alone — `max`
  discards everything but the single peak window per channel.
- If word order / negation matters, put a recurrent or attention layer back over
  the conv outputs — the current trunk is order-free past ~9 chars.
- Keep `EMOJI_EMBED_SIZE = 32` (the revert was right); revisit only after A.

**C. Make eval honest (§2.7, §2.8).**
- Deterministic per-row split: hash of the normalized text, not
  `random.Random(42).shuffle`. Stable as `data.jsonl` grows.
- De-duplicate on the `normalize` key before splitting.
- `TEST_LEN` 900 → ~6,000 (~10%).
- Re-enable `eval/f_loss` + `eval/e_loss`; add **macro-F1** over the 8 feelings.
  Emoji retrieval is now logged as **top-5** hit rate (`{train,eval}/e_acc5`,
  replacing the top-1 `e_acc`) — top-1 0.05 hid whether the neighborhood was
  right, and `rain`→🌧️@3 in the battery says it sometimes is.

### Medium impact

**D. Regularize — but only after B.** Feelings show only a mild gap and
`train/f_loss` is still 0.98, so dropout / heavier weight decay now would hurt.
Once B gives a wider trunk: `WEIGHT_DECAY` → 1e-4, `nn.Dropout(0.1–0.2)` after
the conv stack, `label_smoothing = 0.05` on the feeling CE.

**E. Per-class weighting for the emoji head (§2.6).** Lower priority than
before — the palette is fairly flat (125–170, a few high outliers). Inverse-
frequency `weight=` if A switches the emoji head to CE, or class-balanced
sampling otherwise.

**F. Targeted negation + tail-data guard (§2.9, §2.10).** Explicit negation
texts ("not happy", "wasn't excited", "no longer sad") mapped to sensible
feelings; keep the battery as a release gate. Add an `annotation.ts` guard so
out-of-set feelings stop accumulating in `data.jsonl`.

### Low impact / hygiene

- Reconcile `CLAUDE.md` with the actual code (entry point `train.py`,
  learned-embedding char-CNN → two strided `Conv1d` → global max-pool, no LSTM,
  feeling CE + emoji triplet with 5 table negatives, 300 emojis, `config.py`
  names, real `ExportWrapper` role).
- Fix the "eval feeling loss" comments in `train.py` / `ExportBest` (selection
  is on `eval/f_acc`).
- Reconcile `config.py` `EPOCHS` (30) with the 50-epoch runs in `runs/`.
- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt` and
  reshuffles the split).
- Seed the dataloader workers, or accept the run-to-run noise.

## 4. Suggested order of work

1. **C** — deterministic + larger eval, re-enable the eval-loss scalars,
   macro-F1. Emoji top-5 (`e_acc5`) is now logged. Nothing else here is
   measurable until the eval set is stable and the losses are visible.
2. **A** — fix the emoji loss (in-batch negatives / InfoNCE, or a plain 300-way
   CE head) + loss weighting. This is where the model is worst: last run's
   top-1 `eval/e_acc` 0.051, battery 3/20.
3. **B** — widen `CHANNELS[1]`, `mean ⧺ max` pool, add back an order-sensitive
   layer if negation matters. Unblocks both heads from the 32-channel
   bottleneck.
4. **D** — regularize, once B makes the model large enough to overfit.
5. **E** — emoji class weighting.
6. **F** — negation + annotation guard, once the heads can use it.
- Hygiene (`CLAUDE.md`, stale `train.py` comments, `EPOCHS`) any time.
