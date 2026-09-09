# emojic model

`emojic` maps a short text string to two multi-label label sets and a
colour palette:

- **emojis** — a dynamic-size frequency vocab from `data/labels.json`
  (whatever clears `bun run regen --min-count`).
- **styles** — the fixed closed set of 21 from `tools/data/styles.ts`.
- **palette** — a gradient background (`bg1`, `bg2`) plus a foreground
  `text_color`, as raw sRGB byte offsets.

## Components

### TextEncoder (`model/model.py`)

Char embedding (index 0 = PAD) → a stack of `len(ENCODER_CHANNELS)`
spectral-norm `Conv1d` blocks (`Conv1d(padding=dilation*(k//2),
dilation=…) → LeakyReLU`, constant length, no pooling between blocks).
After each block, PAD positions are masked to `-inf` and a global max
over time is taken; the per-block pooled vectors are concatenated into a
`TEXT_EMBED_SIZE` (= `sum(ENCODER_CHANNELS)`) vector. The forward return
is the raw pooled vector — not L2-normed. The two heads consume it raw;
`ColorGen` and `ColorCritic` L2-norm their text conditioning internally.

### StyleHead / EmojiHead / EmojiEmbedding

`StyleHead` and `EmojiHead` are each `Dropout → Linear(TEXT_EMBED_SIZE →
*_EMBED_SIZE, bias=False)` → a query vector (`STYLE_EMBED_SIZE` 16 /
`EMOJI_EMBED_SIZE` 64).

`StyleHead` keeps its own label-embedding table
(`nn.Embedding(len(STYLES), 16)` + bias,
`logits = proj @ embed.weight.t() + bias`).

The emoji side splits the table out into `EmojiEmbedding` — a shared
module owning `nn.Embedding(len(EMOJIS), 64)` + bias, with
`score(q) = q @ E.t() + bias`. `EmojiHead` returns only the 64-d query
`q_txt`; `emoji_logits = EmojiEmbedding.score(q_txt)`. The same table is
scored by `KWHead` (below), so `E` / `bias` receive gradient from both
`loss/emoji` and `loss/kw`.

Both heads train as retrieval with `model/train.py:lse_infonce` — a
multi-positive Log-Sum-Exp InfoNCE: `logsumexp` over all label logits
minus `logsumexp` over the row's positive logits, scaled by
`INFONCE_TEMP`, averaged over rows with ≥1 positive (emoji rows carrying
no in-vocab emoji drop out). Both losses backprop into the encoder.

### KWHead / FusionHead (`--heads …,fusion`)

A lexical channel over CLDR keywords, fused with `EmojiHead` by a learned
per-row gate.

- **Input** `flex_tf`: a length-`FLEX_N` (dynamic, ≈876) raw soft-TF
  vector over the global keyword vocab `kw_vocab` (`web/public/flex.json`),
  built by the three parity-locked ranker surfaces
  (`tools/data/flexrank.ts` / `model/flexrank.py` /
  `web/src/flexrank.js`). `kw_vocab` = every `df == 1`, single-token,
  6–10-char, corpus-seen CLDR keyword.
- **`KWHead`**: `Dropout(DROPOUT_KW 0.1) → Linear(FLEX_N → 64,
  bias=False)` (weight zero-init) → `q_kw`; `kw_logits =
  EmojiEmbedding.score(q_kw)`, trained with `lse_infonce` (`loss/kw`),
  backprops into `KWHead` + `EmojiEmbedding` (not the trunk).
- **`FusionHead`**: `Linear(TEXT_EMBED_SIZE + FLEX_N → 1)` (weight
  zero-init, bias `+4.0`) → `sigmoid` → per-row gate `a` (≈0.98 at
  init). Fused logits are
  `EmojiEmbedding.score(a * q_txt.detach() + (1-a) * q_kw.detach())` with
  `E` / `bias` detached too, so `loss/fusion` (also `lse_infonce`) trains
  **only** the gate. An untrained gate makes `fusion_logits ≈
  emoji_logits`.

`fusion` requires `emoji`. Stage 1 writes `emoji_embed.pt` (with
`emoji`) and `kw.pt` (with `fusion`).

### ColorCritic (`model/model.py`)

A dual-branch compatibility scorer for a `<text, palette>` pair:

- `text_net`: `L2-norm(text embedding) → CRITIC_TEXT_CHANNELS`
- `color_net`: `rgb_to_oklab(palette) → CRITIC_COLOR_CHANNELS`
- score: `(proj(text_feat) * color_feat).sum(-1)` — a single scalar.

The palette enters the critic in **OKLab** (`model/model.py:rgb_to_oklab`).
Trained with a pairwise BCE loss: `BCE(score(text, real), 1) +
BCE(score(text, negative), 0)`.

### ColorGen (`model/model.py`)

Conditions on the text embedding, mixed with unit-norm noise `z` as
`(1 - Z_WEIGHT) * L2-norm(text_embed) + Z_WEIGHT * z`, through an MLP to
a 9-vector `tanh`-squashed to `[-127.5, 127.5]` — raw sRGB byte offsets
for `bg1`, `bg2`, `text_color`. Output stays in sRGB; the browser
(`web/src/model.js:decodeColors`) adds 127.5, clamps, and hexes.

## Losses at a glance

| Component | Loss |
| --- | --- |
| style, emoji, kw, fusion | multi-positive Log-Sum-Exp InfoNCE (`lse_infonce`) |
| ColorCritic | pairwise BCE, positive + negative `<text, palette>` pairs |
| ColorGen | `Loss_pos` only: `BCE(critic(text, gen(text, z)), 1)` |

`loss/emoji` + `loss/kw` train `EmojiEmbedding`; `loss/kw` also trains
`KWHead`; `loss/fusion` trains only the `FusionHead` gate (all its other
inputs are detached).

Negative `<text, palette>` pairs:

- **Stage 1** — mismatched real palettes from the training set
  (`colors.roll(shift)`; `shift = 1` on validation, random on train).
- **Stage 2** — generator samples (`gen(text, z).detach()`).

## Two-stage training (`model/train.py`)

### Stage 1 — `LitEncoder`

Co-trains `TextEncoder` + the selected heads (`style`, `emoji`,
`critic`, `fusion`; default all four; `fusion` requires `emoji`). The
critic's pairwise BCE loss backprops into the shared encoder; the
`fusion` gate does not (its inputs are detached). Checkpoint +
early-stop on the first available of `MRR/fusion/val` → `F1/val`
(`emoji` + `critic`) → `MRR/e/val` → `MRR/s/val` → `auc/critic/val`
(max). Writes `enc.pt` plus one `.pt` per selected head, plus
`emoji_embed.pt` (with `emoji`) and `kw.pt` (with `fusion`).

Metrics logged: `MRR/s/val`, `MRR/e/val` (full-vocab mean reciprocal
rank, emoji measured only over rows carrying ≥1 emoji), `MRR/kw/val`,
`MRR/fusion/val`, `auc/critic/val` (hand-rolled Mann-Whitney ROC-AUC
over the validation positive/negative critic scores), plus the per-head
losses.

### Stage 2 — `LitColorGAN`

Frozen encoder. `ColorCritic` warm-started from `<--pt>/critic.pt` and
kept training with **generator samples as negatives**; `ColorGen`
trained on `Loss_pos`. Checkpoint + early-stop on `energy/gan/val`
(min) — the OKLab energy distance between real palettes and generator
samples, logged against the split-half real-vs-real `energy/gan/ref`
baseline. Writes `gen.pt`; the stage-2 critic is discarded.

## Artifacts

Saved to the `-o` folder (default `pt/`), each a `{"state_dict",
"meta"}` blob (`model/runmeta.py:save_pt`):

| File | Stage | `meta.stage` |
| --- | --- | --- |
| `enc.pt` | 1 | `enc` |
| `style.pt` | 1 (if `style` in `--heads`) | `enc` |
| `emoji.pt` | 1 (if `emoji` in `--heads`) | `enc` |
| `emoji_embed.pt` | 1 (if `emoji` in `--heads`) | `enc` |
| `kw.pt` | 1 (if `fusion` in `--heads`) | `enc` |
| `fusion.pt` | 1 (if `fusion` in `--heads`) | `enc` |
| `critic.pt` | 1 (if `critic` in `--heads`) | `enc` |
| `gen.pt` | 2 | `gan` |

## CLI

```
train [-h] [enc|gan] [--local] [--heads style,emoji,critic,fusion] [--pt FOLDER] [-o FOLDER]
```

- no positional — Stage 1 (all heads) → Stage 2 → `model/export_onnx.py`
  → `tools/report.py`.
- `enc` — Stage 1 only; `--heads` narrows it (default all four; `fusion`
  requires `emoji`); report, no export.
- `gan` — Stage 2 only; aborts unless `enc.pt` / `critic.pt` /
  `style.pt` / `emoji.pt` / `emoji_embed.pt` / `kw.pt` / `fusion.pt` are
  all in `--pt`; then `gen.pt` → export → report.
- Modal by default; `--local` runs on this machine. `--pt` / `-o` may
  differ from `pt/` only with `--local`, and a non-default `-o` skips
  the `web/public/` export. A dirty git tree always aborts.

`model/export_onnx.py` exports a two-input (`input`, `flex_tf`),
five-output (`style_logits`, `emoji_logits`, `kw_logits`,
`fusion_logits`, `color`) graph; `model/pred.py` and `tools/report.py`
score `EmojiHead` / `KWHead` queries against `EmojiEmbedding` and blend
them through the gate.
