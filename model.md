# Model improvement notes

Based on the latest behavioral report (`report/08-29-20:49.md`), the latest
tensorboard run
(`TIME: 2026-08-29 21:03:34 | MODEL: (20, 30, 40) | TRAIN: lr 0.01 bs 128 gc 1.0 wd 0.0001`),
and a read of `model.py`, `train.py`, `data.py`, `config.py`, `labels.json` and
`test_model.py`.

**The report and the tb run are not the same training — and neither is
finished.** `report/08-29-20:49.md` was produced by the *previous* architecture
(`config.py` at commit `aee1c49`: `CHANNELS = (120, 40)`, two **strided**
`Conv1d` k `(3, 4)` s2, `EPOCHS = 20`) and scored the then-committed `model.pt`.
Commit `2509412` (21:03) then rewrote the model to the **current** source —
three **stride-1** `Conv1d(k=3)`, `CHANNELS = (20, 30, 40)`, `EPOCHS = 500`,
optimizer switched **SGD → Adam** — and started the run above, which is **still
training** (last logged step = epoch **11 / 500**). `model.pt` on disk is
already uncommitted-`M`: `ExportBest` has overwritten it with a current-arch
mid-run checkpoint, so the 20:49 battery no longer describes what is on disk. A
fresh report will be written by `test_model.run()` when the 500-epoch run ends.
Read the battery rows below as "previous arch, converged (20 ep)" and the tb
rows as "current arch, epoch 11, climbing".

**What changed under the model since the previous notes.**

- **Trunk**: `nn.Embedding(38, 16, padding_idx=0)` → transpose →
  `Conv1d(16→20, k3, s1, no bias)` → ReLU → `Conv1d(20→30, k3, s1)` → ReLU →
  `Conv1d(30→40, k3, s1)` → ReLU → `torch.max(dim=-1)` over time → `(B, 40, 1)`.
  The conv stack is built by looping `zip(CHANNELS[:-1], CHANNELS[1:])`, so
  **number of conv layers = `len(CHANNELS)`** and kernel/stride are hardcoded
  (3 / 1), not config. `KERNELS` is gone from `config.py`.
- **No downsampling any more.** Stride is 1 throughout: on `MAX_TEXT_LEN = 42`
  the time axis goes 42 → 40 → 38 → **36**, so the global max pools over **36**
  windows (was ~9 under the strided arch), each with a **7-char** receptive
  field (`1 + 3·(3−1)`). Still no recurrence, no position, no attention.
- **Trunk width = `CHANNELS[-1] = 40`** (was 32 two notes ago, `(256, 32)`). The
  early layers are narrow — 16 → 20 → 30 → 40.
- **Feeling head**: `Conv1d(40→8, k1)`, plain `CrossEntropyLoss`. No
  `label_smoothing`.
- **Emoji head**: `Conv1d(40→32, k1)` → L2-normalize = anchor `q`; learned
  `emoji_embed = nn.Embedding(300, 32)`, rows L2-normalized.
  `nn.TripletMarginLoss(margin=0.5)`; each row draws **`EMOJI_NEGATIVES = 5`**
  negatives, each still a **random other row of the same learned table**
  (`emoji_embed[(target + rand 1..N-1) % N]`), tiled to `(B*5, D)` and meaned.
  Prediction = nearest `emoji_embed` row to `q` (`torch.cdist` in test,
  `q @ emoji_embed.T` in eval / `ExportWrapper`). **Unchanged in substance since
  the last two notes — still the #1 problem (§2.1).**
- **Total loss** = unweighted sum `loss_feeling + loss_emoji` (CE ≈ 1.5 +
  triplet ≈ 0.3).
- **Optimizer: Adam** (`lr 0.01`, `wd 1e-4`). The `optim.SGD(...)` branch is
  commented out in `configure_optimizers`. Previous notes described SGD `lr 0.1`
  `wd 2e-4` — both stale.
- `config.py`: `LR` 0.1 → **0.01**; `WEIGHT_DECAY` 2e-4 → **1e-4**; `EPOCHS`
  30/50 → **500**; `EVAL_EPOCHS = 2`; `CHAR_EMBED_SIZE = 16`,
  `EMOJI_EMBED_SIZE = 32`, `TRIPLET_MARGIN = 0.5`, `EMOJI_NEGATIVES = 5`
  unchanged. `CONFIG_NAME` lost its `DATA:` / `cs` fields — hence the bare
  `MODEL: (20, 30, 40)` run name.
- **Params ≈ 18.2k** (was 56.6k). The **`emoji_embed` 300×32 table is 9.6k =
  52.7%** of the model; the entire conv feature extractor is **6.36k = 34.9%**
  (`conv3` 30×40×3 = 3.6k is the largest single tensor). Both heads are tiny
  (feeling 328, emoji proj 1,312). The model is mostly a lookup table with a
  6k-param front end.
- **Logging**: tb has `train/{f_acc, e_acc5, f_loss, e_loss}` and
  `eval/{f_acc, e_acc5}`. The emoji metric is **top-5 hit rate** (`e_acc5`);
  there is no top-1 emoji scalar. **`eval/f_loss` and `eval/e_loss` are still
  commented out** in `validation_step` — the train/eval *loss* gap still cannot
  be read, and the run is now 25× longer.
- `ExportBest` selects `model.pt` on **`eval/f_acc`** (max). Its docstring and
  `train.py`'s module docstring still say "by eval feeling loss" — stale.
- `labels.json`: 8 feelings (includes **Love**); **300** emojis.
- `normalize` carries the run-collapse `re.sub(r'(.)\1{2,}', r'\1\1', text)` —
  must stay byte-identical with `docs/app.js`. `CHARS` still has no digits;
  `VOCAB_SIZE = 38`.
- **Corpus**: `data.jsonl` **67,439** rows; after `data.py:read()` (feeling ∈ 8,
  emoji ∈ 300, normalized len ≤ 42) **59,881** trainable (~7,558 dropped —
  almost all for an out-of-palette emoji: 857 distinct emojis raw, 300 kept;
  plus **14** out-of-set feeling rows: Annoyed 5, Confused 4, Frustrated 2,
  Hopeful/Amused/Relieved 1). Split = `random.Random(42).shuffle(read())[:900]`
  for eval, rest train → **58,981 train / 900 eval**.

**Headline.** Same two stories as the last three notes.
**Feelings**: real signal, not yet plateaued *in this run* because the run is
young — `eval/f_acc` 0.400 → **0.462** at epoch 11 (vs a **~0.22**
always-Neutral baseline), `train/f_acc` **0.475**, so the train/eval gap is only
~1.3 pt (no overfitting visible yet at epoch 11). The name battery (previous
arch) was **6/8** (`angry`→Neutral, `love`→Neutral).
**Emoji**: still no text→emoji map. `eval/e_acc5` **0.167** over 300 classes
(~10× the 5/300 chance rate for top-5), the triplet loss slides from 0.394 to
**0.291** — well inside the 0.5 margin — without `q` ever having to resolve
which emoji a *text* means, and the cue battery (previous arch) was **1/20**
top-1, 5/20 top-3, 8/20 top-5. Raising `EMOJI_NEGATIVES` to 5 changed nothing
because all 5 negatives are rows of the same learned table (§2.1).

## 1. Where the model stands

| Battery | Result | Read |
| --- | --- | --- |
| Feelings by name (report, **prev arch**, 20 ep) | **6/8 (75%)** | ✅ `happy`, `excited`, `calm`, `sad`, `anxious`, `neutral`. ❌ `angry`→Neutral, `love`→Neutral. 8-point check, noisy — prior report was 7/8; treat 6–7/8 as the band. `love` fails as a model miss, not a data gap (Love has 6,455 trainable rows). |
| Negations `not <feeling>` (report, **prev arch**) | **6/8** avoided the named feeling; **2/6** hit the expected opposite; **4/8** full pass ("Neg Feeling Score") | `not happy`→Sad and `not calm`→Anxious are the only two that both avoid *and* land the `NEGATION_EXPECTED` opposite. `not neutral`/`not love`→Sad "pass" only because no opposite is defined. `not sad`→Sad and `not anxious`→Anxious don't even avoid. Negation is a near no-op; 4/8 is noise. |
| Eval feeling acc — tb, **current arch, IN PROGRESS** | **0.462** at epoch **11 / 500**, rising monotonically from 0.400 | vs a **~0.22** majority-class baseline (always-Neutral: 13,353 / 59,881 filtered rows). Only 6 evals logged so far (every 2 epochs); no plateau yet. `ExportBest` ships the max-`eval/f_acc` checkpoint. |
| Eval emoji top-5 — tb, **current arch, IN PROGRESS** | `e_acc5` **0.167** at epoch 11, from 0.142 | Top-5 nearest-embedding retrieval over 300 classes; chance = 5/300 = **0.0167**, so ~10× chance but useless in absolute terms. **No top-1 emoji scalar is logged** — the previous top-1 `e_acc` was replaced by this. |
| Emoji cue battery (report, **prev arch**) | top-1 **1/20**, top-3 **5/20**, top-5 **8/20** | Only `nervous`→😰 at rank 1. `furious`→😡, `heartbroken`→💔, `rain`→🌧️, `whatever`→🙄 in the top 3; `crying`→😢, `coffee`→☕, `terrified`→😱 by rank 5. Positive/neutral cues collapse to hub vectors: ✅ is top-1 for 5 of 20 (`party`, `coffee`, `grateful`, `meditate`, `sparkle`), 😐 for 4 (`thinking`, `birthday`, `adore`, `down`); swimming emojis (🏊 / 🏊‍♀️) pollute the tails. |
| Train losses (tb, current arch) | `f_loss` 1.842 → **1.468** at epoch 11 (`ln 8` = 2.079; still falling steeply); `e_loss` 0.394 → **0.291** (triplet, margin 0.5) | `f_loss` far above 0 with `train/f_acc` 0.475 ≈ `eval/f_acc` 0.462 ⇒ heavy underfitting / early in training, no memorisation. `e_loss` ≈ 0.6× margin and dropping ⇒ the 300 table rows are separating on their own; no useful text→emoji map. |

## 2. Root causes

### 2.1 The emoji triplet signal is degenerate (biggest, unchanged three notes running)
`training_step` builds every negative as
`emoji_embed[(target_emoji + offset) % N]` — a **random other row of the same
learned table**, sampled 5× per anchor. There is no contrast against other
*texts* in the batch, no hard-negative mining, and the "negative" is a parameter
the optimizer also controls. So `e_loss` falls to ~0.29 just from the 300 table
rows drifting apart, while `q` (a 32-d projection of a lossy 40-channel
max-pool) never has to resolve which emoji a given *text* means. Result:
`eval/e_acc5` 0.167, battery 1/20 top-1. This is a loss-design problem, not a
capacity one, and `EMOJI_NEGATIVES` 1 → 5 confirmed it — more negatives from the
same source did nothing.

### 2.2 The model is 53% lookup table, 35% feature extractor
`emoji_embed` (300×32 = 9,600 params) is **52.7%** of the 18.2k-param model; the
whole conv stack is **6.36k**. Even if §2.1 were fixed, a 6k-param feature
extractor feeding a 40-channel bottleneck cannot produce a 32-d text embedding
discriminative enough to retrieve one of 300 classes. Either drop the table for
a plain CE head (§3.A) or the feature extractor needs multiples of its current
budget.

### 2.3 Global max-pool over 36 bag-of-7-gram windows = order-free
`Embedding → 3× Conv1d(k3, s1) → max over time`. Each of the 36 surviving steps
sees 7 input characters; the max keeps only the single most-activating window
per channel. No recurrence, no position, no long-range composition. Negation,
word order and scope past ~7 chars are structurally invisible — hence the
near-zero negation battery and the positive/neutral cue collapse to ✅ / 😐.
Removing the strides widened the window count 9 → 36 but did not add any
order sensitivity.

### 2.4 The trunk is 40 channels feeding both heads, and narrow early
`CHANNELS = (20, 30, 40)`: a 16-d char embedding is immediately squeezed to 20
channels, and the same 40 numbers (after a lossy max-pool) must carry an 8-way
feeling decision **and** a 300-way emoji retrieval. Feelings will plateau well
short of ceiling; a 300-way signal does not fit through 40 d.

### 2.5 Unweighted sum of two differently-scaled losses
`return loss_feeling + loss_emoji` — CE (~1.5) plus triplet (~0.3), no weight,
no normalization. The feeling CE dominates the gradient; the emoji head gets the
leftover capacity of an already-tiny trunk.

### 2.6 Eval losses are still not logged — and the run is now 500 epochs
`validation_step` comments out `eval/f_loss` and `eval/e_loss`. With `EPOCHS`
20/50 → **500** and Adam, this is now the single biggest blind spot: there is no
way to see where eval loss bottoms out or when overfitting starts, and
`ExportBest`'s docstring still claims selection is on eval loss (it is on
`eval/f_acc`). `train/f_acc` is logged, so the *accuracy* gap is visible — but
at epoch 11 it is ~1 pt and uninformative.

### 2.7 Eval split is 900 random rows, non-stationary, no dedup
`TEST_LEN = 900` is 1.5% of 59,881. `data.py:split()` does
`random.Random(42).shuffle(read())` then takes the first 900; `data.jsonl` is
append-only (`add-samples` keeps growing it), so **every data add reshuffles the
holdout** and runs across an add are not comparable. No de-duplication on the
`normalize` key before the split, so near-identical generated texts can straddle
it. `eval/f_acc` jitters ~±0.015 between adjacent evals.

### 2.8 Class balance: solved for feelings, mild for emoji
After `read()`, feelings span Neutral **13,353** … Happy 6,958 … Love **6,455**
— **2.07×** head-to-tail, and Love is fully populated. `love`→Neutral is a
capacity miss, not a data gap. Emoji palette counts are tight: min **125** (🪠),
median **150**, max **1,099** (😤), with a short high tail (😤 1,099, 🎉 1,025,
😌 969, 😠 897, 😔 734, ☕ 699). **0 of the 300 palette emojis have zero rows.**
Emoji imbalance is not the story — §2.1 is.

### 2.9 Negation is unsupervised and OOD
`test_model.py` feeds bare `not <feeling>` (1–2 tokens) while training texts are
informal sentences. Nothing in the loss rewards `not`; `NEGATION_EXPECTED`
covers 6 feelings. Keep the battery as a release gate, not a metric that moves
on its own.

### 2.10 Minor / hygiene
- `ExportBest`'s docstring and `train.py`'s module docstring say the checkpoint
  is chosen "by eval feeling loss"; the code selects on `eval/f_acc` and eval
  loss is not logged.
- `data.jsonl` carries **14** out-of-set feeling rows (Annoyed 5, Confused 4,
  Frustrated 2, Hopeful/Amused/Relieved 1). `read()` drops them silently; an
  `annotation.ts` guard would stop the drift.
- `CHARS` still has no digits; `VOCAB_SIZE = 38`.
- `num_workers=4` without per-worker seeding leaves run-to-run noise despite
  `pl.seed_everything(0, workers=True)`.
- `SGD` branch left commented in `configure_optimizers` — dead code; the active
  path is Adam.
- The emoji cue battery is `test_model.py`'s own inline 20-pair `EMOJI_CUES`;
  widen it once the emoji head works.
- `CLAUDE.md` is stale on nearly every technical point: it describes a
  `pack_padded_sequence` LSTM with two cross-entropy heads, `main.py` as the
  entry point, an "80-emoji palette", `gen_labels.ts` as "top 200", and
  `config.py` names `EMBED_SIZE` / `H_SIZE` / `NUM_LAYERS`. Actual: learned
  `nn.Embedding` char-CNN → **three stride-1 `Conv1d(k=3)`** → global max-pool
  (no LSTM), feeling CE + emoji **triplet** (5 table negatives), `train.py`
  (Lightning) entry, **Adam**, **300** emojis, `config.py` names `CHANNELS` /
  `CHAR_EMBED_SIZE` / `EMOJI_EMBED_SIZE` / `TRIPLET_MARGIN` / `EMOJI_NEGATIVES`.
  `ExportWrapper` collapses the emoji embedding head to `emoji_logits`; it does
  not touch `pack_padded_sequence`.

## 3. Recommendations (prioritised)

### Done since the previous notes
- **Architecture rewritten again**: strided 2-conv `(120, 40)` k`(3,4)` →
  **three stride-1 `Conv1d(k=3)`** `(20, 30, 40)`; `KERNELS` removed from
  `config.py`; layer count now follows `len(CHANNELS)`. Pooling window count
  9 → 36; receptive field 7 chars.
- **Optimizer SGD → Adam**; `LR` 0.1 → 0.01; `WEIGHT_DECAY` 2e-4 → 1e-4.
- `EPOCHS` → **500**, `EVAL_EPOCHS = 2`.
- Emoji metric is top-5 (`{train,eval}/e_acc5`); top-1 `e_acc` removed.
- Feeling class balance holds (Love fully populated, 6,455 rows).
- **Not done, still #1**: the emoji loss (rec A). **Not done**: re-enable
  `eval/f_loss` + `eval/e_loss` (rec C) — now more urgent at 500 epochs.
  **Not done**: deterministic split (rec C). **Not done**: loss weighting.

### High impact

**A. Fix the emoji training signal (§2.1, §2.2, §2.5) — still the #1 change.**
- Replace the table-sampled triplet negatives with **in-batch negatives**: an
  InfoNCE / cross-entropy over `q @ emoji_embed.T` (temperature-scaled) gives
  every row `B−1` real *text* contrasts per step and is the standard text→label
  retrieval loss. Or — simplest, and given §2.2 probably a higher ceiling —
  drop `emoji` proj + `emoji_embed` for a plain `Conv1d(40→300, k=1)` CE head.
  300-way CE is well-trodden, frees the 9.6k table budget for the trunk, and
  `ExportWrapper` collapses to a no-op.
- Whatever the head, **weight the two losses** (`loss_feeling + λ·loss_emoji`,
  or normalize their scales) so the emoji gradient isn't swamped by the feeling
  CE.

**B. Make eval honest (§2.6, §2.7) — do this first, it's cheap and the run is
now 25× longer.**
- Re-enable `eval/f_loss` + `eval/e_loss` in `validation_step`. Without them a
  500-epoch Adam run has no visible overfitting signal and `ExportBest`'s own
  docstring is a lie.
- Deterministic per-row split: hash of the normalized text, not
  `random.Random(42).shuffle`. Stable as `data.jsonl` grows.
- De-duplicate on the `normalize` key before splitting.
- `TEST_LEN` 900 → ~6,000 (~10%).
- Add **macro-F1** over the 8 feelings.

**C. Give the representation room (§2.2, §2.3, §2.4).**
- Widen the trunk: the early 20-channel layer right after a 16-d embedding is a
  hard bottleneck. Try `CHANNELS = (64, 96, 128)` (or similar) so `CHANNELS[-1]`
  is 96–128, not 40.
- Pool `mean ⧺ max` over the conv time axis instead of `max` alone — `max`
  discards everything but the single peak window per channel.
- If word order / negation matters, put a recurrent or attention layer back over
  the conv outputs — the current trunk is order-free past ~7 chars.

### Medium impact

**D. Regularize — but only after C.** At epoch 11 the model is deeply
underfit (`train/f_loss` 1.47, gap ~1 pt), so dropout / heavier weight decay now
would hurt. Once C widens the trunk and B makes overfitting visible:
`WEIGHT_DECAY` sweep around 1e-4, `nn.Dropout(0.1–0.2)` after the conv stack,
`label_smoothing = 0.05` on the feeling CE. Also revisit `LR` — 0.01 Adam may be
low; watch the re-enabled `eval/f_loss` before deciding.

**E. Per-class weighting for the emoji head (§2.8).** Low priority — the palette
is fairly flat (125–170, a few high outliers). Inverse-frequency `weight=` if A
switches the emoji head to CE, or class-balanced sampling otherwise.

**F. Targeted negation + tail-data guard (§2.9, §2.10).** Explicit negation
texts ("not happy", "wasn't excited", "no longer sad") mapped to sensible
feelings; keep the battery as a release gate. Add an `annotation.ts` guard so
out-of-set feelings stop accumulating in `data.jsonl`.

### Low impact / hygiene

- Reconcile `CLAUDE.md` with the actual code (entry point `train.py`,
  learned-embedding char-CNN → three stride-1 `Conv1d(k=3)` → global max-pool,
  no LSTM, feeling CE + emoji triplet with 5 table negatives, Adam, 300 emojis,
  `config.py` names, real `ExportWrapper` role).
- Fix the "eval feeling loss" comments in `train.py` / `ExportBest` (selection
  is on `eval/f_acc`).
- Delete the commented-out `SGD` branch in `configure_optimizers`.
- Add `0123456789` to `CHARS` (retrain + re-export; invalidates `model.pt` and
  reshuffles the split).
- Seed the dataloader workers, or accept the run-to-run noise.

## 4. Suggested order of work

1. **B** — re-enable the eval-loss scalars, deterministic + larger eval,
   macro-F1. Nothing else is measurable until eval is stable and the losses are
   visible, and a 500-epoch run without them is flying blind.
2. **A** — fix the emoji loss (in-batch negatives / InfoNCE, or a plain 300-way
   CE head) + loss weighting. This is where the model is worst: `eval/e_acc5`
   0.167 (top-5), battery 1/20 top-1.
3. **C** — widen `CHANNELS`, `mean ⧺ max` pool, add an order-sensitive layer if
   negation matters. Unblocks both heads from the 40-channel / 6k-param
   bottleneck.
4. **D** — regularize and retune `LR`, once C makes the model large enough to
   overfit and B makes it visible.
5. **E** — emoji class weighting.
6. **F** — negation + annotation guard, once the heads can use it.
- Hygiene (`CLAUDE.md`, stale `train.py` comments, dead SGD branch, `EPOCHS`)
  any time.
