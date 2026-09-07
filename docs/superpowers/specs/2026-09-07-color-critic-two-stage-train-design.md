# Contrastive color critic + two-stage `train` CLI

## Goal

Replace the adversarial-BCE color GAN and the sprawling
`train <emoji|style|task|gan|all|color-exp|critic>` CLI with:

- a **contrastive `ColorCritic`** (dual-branch dot-product compatibility
  scorer) trained with a pairwise BCE loss,
- a **generator** trained on the positive term of that loss only
  (`Loss_pos`),
- a clean **two-stage** training flow behind one `train` command:
  `train [enc|gan] [--local] [--heads emoji,style,critic] [--pt FOLDER] [-o FOLDER]`.

The text encoder and the two retrieval heads (`StyleHead`, `EmojiHead`)
are architecturally unchanged: both remain `text embedding -> label
retrieval` heads, independent of each other.

## Non-goals

- No change to `TextEncoder`, `StyleHead`, `EmojiHead`, `normalize`,
  char vocab, or the `.pt` blob format.
- No head chaining (an earlier draft of the request had a typo:
  "EmojiHead: style embedding -> emoji embedding"; the head stays on the
  text embedding).
- No change to `web/`, `model/pred.py`, `tools/report.py`,
  `model/export_onnx.py` internals, `model/runmeta.py`.
- Arbitrary `--pt` / `-o` folders on the Modal path are out of scope
  (Modal always uses `pt/` in-container).

## Architecture

```
TextEncoder : text ids           -> text embedding      (416-d, unchanged)
StyleHead   : text embedding      -> style logits        (retrieval, LSE-InfoNCE)
EmojiHead   : text embedding      -> emoji logits        (retrieval, LSE-InfoNCE)
ColorCritic : (text embedding, palette) -> scalar score  (contrastive, pairwise BCE)
ColorGen    : text embedding + noise    -> 9-vec palette (generative, Loss_pos)
```

`ColorCritic` (renamed from `ColorDsc`, body unchanged):

- `text_net`  : `normalize(text_embedding)` -> `CRITIC_TEXT_CHANNELS`
- `color_net` : `rgb_to_oklab(palette)`     -> `CRITIC_COLOR_CHANNELS`
- `proj`      : `CRITIC_TEXT_CHANNELS[-1]`  -> `CRITIC_COLOR_CHANNELS[-1]`
- score       : `(proj(t) * c).sum(-1, keepdim=True)`

The palette enters the critic in **OKLab**; `ColorGen` output stays in
sRGB byte-offset space (`tanh * 127.5`).

## Losses

| Component | Loss |
|---|---|
| style, emoji | `lse_infonce` (multi-positive Log-Sum-Exp InfoNCE), `INFONCE_TEMP` |
| ColorCritic | pairwise BCE: `BCE(score(text, real_palette), 1) + BCE(score(text, neg_palette), 0)` |
| ColorGen | `Loss_pos` only: `BCE(score(text, gen(text, z)), 1)` |

**Negative `<text, palette>` pairs:**

- **Stage 1** (encoder training): negatives are mismatched real
  palettes from the training set — `colors.roll(shift, dims=0)`, with
  `shift = 1` on validation and a random `1..B-1` on each training
  batch.
- **Stage 2** (GAN training): negatives are generator samples
  (`gen(text, z).detach()`).

The critic loss backprops into the shared `TextEncoder` in stage 1.
In stage 2 the encoder is frozen and the critic is a throwaway model.

## Two-stage training

### Stage 1 — `LitEncoder` (renamed from `LitTask`)

- `__init__(heads: tuple[str, ...])`, `heads` a subset of
  `{"style", "emoji", "critic"}`. Always builds `TextEncoder`; builds
  `StyleHead` / `EmojiHead` / `ColorCritic` only for selected heads.
- Per-step, for each selected head:
  - **style**: `lse_infonce(style_logits, style, INFONCE_TEMP)`;
    log `loss/s/{split}`, `MRR/s/{split}`.
  - **emoji**: `lse_infonce` over rows with >=1 emoji;
    log `loss/e/{split}`, `MRR/e/{split}` (emoji MRR measured only over
    those rows).
  - **critic**: pairwise BCE as above; log `loss/critic/{split}`,
    `acc/critic/{split}`, and on validation `auc/critic/val` — a
    hand-rolled Mann-Whitney ROC-AUC over the concatenated positive /
    negative score vectors:
    `AUC = (sum(rank(pos)) - n_p*(n_p+1)/2) / (n_p * n_n)`.
    No `torchmetrics`.
- Total loss = sum of the selected head losses.
- `configure_optimizers`: `Adam(enc.params + selected head params, lr=LR)`.
- **Checkpoint / early-stop** (`ModelCheckpoint` + `EarlyStopping`,
  `mode="max"`), monitor by priority:
  1. `MRR/e/val` if `"emoji"` in `heads`
  2. else `MRR/s/val` if `"style"` in `heads`
  3. else `auc/critic/val`
- On completion, reload the best checkpoint and save `enc.pt` plus one
  `.pt` per selected head (`style.pt` / `emoji.pt` / `critic.pt`) to the
  output folder via `save_pt(..., stage="enc")`.

### Stage 2 — `LitColorGAN` (kept, trimmed)

- `__init__(enc, critic)`: `enc` frozen (`.eval().requires_grad_(False)`);
  `self.gen = ColorGen()`; `self.tst = critic` warm-started from
  `--pt/critic.pt`.
- `training_step` (manual optimization, unchanged in spirit):
  - `cond = enc(text).detach()`, `fake = gen(cond)`.
  - critic step: `loss_tst = BCE(critic(cond, colors), 1) +
    BCE(critic(cond, fake.detach()), 0)`; clip `GRAD_CLIP_CRITIC`;
    `opt_tst.step()`.
  - generator step: `loss_gen = BCE(critic(cond, fake), 1)`; clip
    `GRAD_CLIP_GEN`; `opt_gen.step()`.
  - log `loss/gan/tst`, `loss/gan/gen`.
- Validation unchanged: accumulate text + real palettes, compute
  `energy/gan/val` = OKLab energy distance between real palettes and
  `gen(enc(text), z_bank)`, and `energy/gan/ref` = split-half real-vs-real
  baseline.
- **Checkpoint / early-stop**: `energy/gan/val`, `mode="min"`.
- Saves `gen.pt` only (via `save_pt(..., stage="gan")`); the stage-2
  critic is discarded.

### Deleted

`ColorHead`, `LitColorExp`, `LitCriticExp`, `_run_color_exp`,
`_run_critic_exp`, the `Model` enum, `TASK_HEADS`, the
`modal run model/train.py::main` `local_entrypoint`, the `--fetch-only`
branch, `--cpu` / `--memory` (box size hardcoded to `CPU=16` /
`MEMORY_MIB=16384`). `_retrieve_and_cleanup` and `_run_report_local`
stay (still needed after a blocking Modal run).

## CLI

```
train [-h] [enc|gan] [--local] [--heads emoji,style,critic] [--pt FOLDER] [-o FOLDER]
```

| Form | Behavior |
|---|---|
| `train` (no positional) | Stage 1 with all three heads -> reload best ckpt -> Stage 2 (critic warm-started from the just-written `-o/critic.pt`) -> `export()` -> report. Modal unless `--local`. |
| `train enc` | Stage 1 only. `--heads` selects the subset (default: all three). Saves `enc.pt` + selected heads to `-o`. Report. No export. |
| `train gan` | Stage 2 only. Requires `enc.pt`, `critic.pt`, `style.pt`, `emoji.pt` in `--pt` (abort if any missing). Saves `gen.pt` to `-o` -> `export()` -> report. |

Flags:

- `--local` — run on this machine instead of Modal. Valid with any form.
- `--heads LIST` — comma-separated, subset of `{emoji, style, critic}`.
  Only valid with `enc`. Any unrecognized name aborts. Given with `gan`
  or the no-arg form aborts.
- `--pt FOLDER` — input `.pt` folder, default `pt/`.
- `-o` / `--output FOLDER` — output `.pt` folder, default `pt/`.
- A non-default `--pt` or `-o` with a Modal run aborts (Modal always
  uses `pt/` in-container). `export()` always reads/writes the `pt/`
  constants from `files.py`; a non-default `-o` therefore writes `.pt`
  there but **skips** the `web/public/` refresh — including for
  `train gan`, whose export step only runs when `-o` is the default
  `pt/`.
- Dirty git tree aborts every form (`require_clean_tree`, no override).
- Invalid flag combinations abort with a Typer error.

### Modal path

`train_remote(stage, heads, git_sha, run_time, enc_bytes, style_bytes,
emoji_bytes, critic_bytes)`:

- For `gan`: uploads all four `.pt` (read from local `pt/`) into the
  container's `pt/` before training; aborts locally first if any is
  missing.
- For `enc` / full: uploads none.
- In-container: `python model/train.py <stage> [--heads ...] --local`
  with `EMOJIC_SKIP_REPORT=1`; forwards a TensorBoard tunnel.
- Stashes `pt/`, `runs/`, `web/public/`, `report/` to the volume;
  `_dispatch` fetches them back and runs `tools/report.py` locally once
  artifacts have landed.

## Artifacts

Saved to `-o` (default `pt/`), each a `{"state_dict", "meta"}` blob via
`save_pt`:

| File | Written by | `meta.stage` |
|---|---|---|
| `enc.pt` | stage 1 | `enc` |
| `style.pt` | stage 1 (if `style` selected) | `enc` |
| `emoji.pt` | stage 1 (if `emoji` selected) | `enc` |
| `critic.pt` | stage 1 (if `critic` selected) | `enc` |
| `gen.pt` | stage 2 | `gan` |

`files.py`: `TST_PT` -> `CRITIC_PT = "pt/critic.pt"`; delete `EXP_DIR`,
`EXP_ENC_PT`, `EXP_GAN_PT`, and the Python `COLORS_JSONL` constant
(`files.ts` keeps its `COLORS_JSONL` for the live `extract-colors.ts`).

## Eval metrics

| Component | Metric | Role |
|---|---|---|
| style | `MRR/s/val` (full-vocab MRR) | logged; stage-1 monitor fallback |
| emoji | `MRR/e/val` (full-vocab MRR, rows with >=1 emoji) | logged; stage-1 monitor (primary) |
| ColorCritic | `auc/critic/val` (Mann-Whitney ROC-AUC) | logged; stage-1 monitor fallback (critic-only runs) |
| ColorGen | `energy/gan/val` vs `energy/gan/ref` | stage-2 monitor / early-stop / checkpoint |

## Files touched

- `model/model.py` — delete `ColorHead`; rename `ColorDsc` -> `ColorCritic`.
- `model/config.py` — no hyperparameter changes (names already fit).
- `files.py` — `CRITIC_PT`; delete `EXP_*` + `COLORS_JSONL`.
- `model/train.py` — full rewrite of the CLI + stage plumbing; rename
  `LitTask` -> `LitEncoder`; add ROC-AUC + critic branch to stage 1;
  trim `LitColorGAN`; delete exp code, `Model` enum, `main`,
  `--fetch-only`, `--cpu`/`--memory`.
- `CLAUDE.md` — rewrite the `model/train.py` bullet, `pt/` bullet
  (`tst.pt` -> `critic.pt`), "Full run" / verification lines, Conventions
  color paragraph.
- `docs/model.md` — create fresh (prose description of the model +
  training).

## Verification

- `uv run ruff check .` && `uv run ruff format --check .`
- `uv run python model/test_runmeta.py`
- `train enc --local --heads style,emoji,critic` — watch `MRR/e/val`,
  `MRR/s/val`, `auc/critic/val` in TensorBoard; confirm `enc.pt` /
  `style.pt` / `emoji.pt` / `critic.pt` written.
- `train gan --local` — confirm it aborts cleanly when a required `.pt`
  is absent, otherwise watch `energy/gan/val` vs `energy/gan/ref`,
  confirm `gen.pt` + `web/public/` refresh + report.
- `train --local` — full two-stage run end to end.
- `uv run python model/pred.py --pt pt` — `data/pred.jsonl` spot-check.
- `uv run python tools/report.py --pt pt` — full run report.
- `train enc --heads bogus` / `train gan --heads style` / `train --pt x`
  (no `--local`) — all abort with a clear message.
```
