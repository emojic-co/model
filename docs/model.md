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

### StyleHead / EmojiHead

`StyleHead` and `EmojiHead` are each `Dropout → Linear(TEXT_EMBED_SIZE →
*_EMBED_SIZE, bias=False)` → a query vector (`STYLE_EMBED_SIZE` 16 /
`EMOJI_EMBED_SIZE` 64), scored against a label-embedding table owned by
the same module (`nn.Embedding(len(labels), *_EMBED_SIZE)` + bias,
`logits = proj @ embed.weight.t() + bias`). `EmojiHead.score(q)` exposes
the table-scoring step on its own for callers (`tools/report.py`) that
need to probe a raw query vector.

Both heads train as retrieval with `model/train.py:lse_infonce` — a
multi-positive Log-Sum-Exp InfoNCE: `logsumexp` over all label logits
minus `logsumexp` over the row's positive logits, scaled by
`INFONCE_TEMP`, averaged over rows with ≥1 positive (emoji rows carrying
no in-vocab emoji drop out). Both losses backprop into the encoder.

CLDR/EmojiLib keywords and terms (`data/keywords.jsonl`, `data/terms.jsonl`
— see `tools/data/keywords.ts`) are mixed directly into `emoji`/`style`
training as regular text rows (`model/data.py:EmojiDataset(mix_sources=True)`),
each source (`SAMPLING_SOURCES` in `model/config.py` — file, TB metric, and
metric goal per source) sampled at a shared base rate that decays toward a
shared floor as that source's own metric approaches its goal (updated in
`LitEncoder.on_train_epoch_start` — see `docs/search.md`) rather than through
a separate lexical head — there is no `KWHead`/`FusionHead`/gate in the
model, and no keyword lookup or fusion step in the web app; `EmojiHead` is
queried directly on whatever text the user typed.

### ColorCritic (`model/model.py`)

A dual-branch compatibility scorer for a `<text, palette>` pair:

- `text_net`: `L2-norm(text embedding) → CRITIC_TEXT_CHANNELS`
- `color_net`: `rgb_to_oklab(palette) → CRITIC_COLOR_CHANNELS`
- score: `(proj(text_feat) * color_feat).sum(-1)` — a single scalar.

The palette enters the critic in **OKLab** (`model/model.py:rgb_to_oklab`).
Returns a `(color_score, cond_score)` pair — a color-only branch and a
text-conditioned branch, combined via `LOSS_WEIGHT_COND_COLOR` in stage
2's hinge loss. Trained from scratch each `gan` run — there is no
stage-1 warm start.

### ColorGen (`model/model.py`)

Conditions on the text embedding, mixed with unit-norm noise `z` as
`(1 - Z_WEIGHT) * L2-norm(text_embed) + Z_WEIGHT * z`, through an MLP to
a 9-vector `tanh`-squashed to `[-127.5, 127.5]` — raw sRGB byte offsets
for `bg1`, `bg2`, `text_color`. Output stays in sRGB; the browser
(`web/src/model.js:decodeColors`) adds 127.5, clamps, and hexes.

## Losses at a glance

| Component | Loss |
| --- | --- |
| style, emoji | multi-positive Log-Sum-Exp InfoNCE (`lse_infonce`) |
| ColorCritic | hinge loss on `(color_score, cond_score)`, weighted by `LOSS_WEIGHT_COND_COLOR`; real palettes vs. generator samples |
| ColorGen | negative critic score (both branches) blended with OKLab `energy_distance` via `GAN_LOSS_ENERGY` |

`loss/emoji` trains `EmojiHead`; `loss/style` trains `StyleHead`.

The critic's only negatives are generator samples (`gen(text,
z).detach()`) — there is no stage-1 phase training it against mismatched
real palettes.

## Two-stage training (`model/train.py`)

### Stage 1 — `LitEncoder`

Co-trains `TextEncoder` + the selected heads (`style`, `emoji`, `lang`;
default all three; `ColorCritic` is not a stage-1 head). Checkpoint +
early-stop on the first available of `F1/val` (`emoji` + `style`) →
`MRR/e/val` → `MRR/s/val` → `acc/lang/val` (max). Writes `enc.pt` plus
one `.pt` per selected head.

Metrics logged: `MRR/s/val`, `MRR/e/val` (full-vocab mean reciprocal
rank, emoji measured only over rows carrying ≥1 emoji), `acc/lang/val`,
plus the per-head losses.

### Stage 2 — `LitColorGAN`

Frozen encoder. `ColorCritic` is built fresh (`ColorCritic()`) — no
warm start, so `gan` runs against any encoder checkpoint — and trained
opposite `ColorGen` with **generator samples as negatives** throughout.
Checkpoint + early-stop on `energy/gan/val` (min) — the OKLab energy
distance between real palettes and generator samples, logged against
the split-half real-vs-real `energy/gan/ref` baseline. Writes `gen.pt`;
the stage-2 critic is discarded.

## Artifacts

Saved to `pt/`, each a `{"state_dict", "meta"}` blob
(`model/runmeta.py:save_pt`):

| File | Stage | `meta.stage` |
| --- | --- | --- |
| `enc.pt` | 1 | `enc` |
| `style.pt` | 1 | `enc` |
| `emoji.pt` | 1 | `enc` |
| `gen.pt` | 2 | `gan` |

## CLI

```
train [-h] [gan] [--local]
```

- no positional — Stage 1 (all heads) → Stage 2 → `model/export_onnx.py`
  → `tools/report.py`.
- `gan` — Stage 2 only; aborts unless `enc.pt` / `style.pt` / `emoji.pt`
  are all in `pt/`; then `gen.pt` → export → report.
- Modal by default; `--local` runs on this machine. A dirty git tree
  always aborts.

`model/export_onnx.py` exports a single-input (`input`), three-output
(`style_logits`, `emoji_logits`, `color`) graph; `model/pred.py` and
`tools/report.py` score `EmojiHead` directly, with no fusion/blending
step.
