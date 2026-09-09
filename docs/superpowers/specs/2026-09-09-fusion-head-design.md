# FusionHead — design

## Goal

Add a learned head that re-ranks `EmojiHead`'s per-label logits using the CLDR
keyword-retrieval signal already written to `data/train.jsonl` / `data/eval.jsonl`
by `regen` (`flexsearch` / `flexq` fields). The head is trained end to end in
stage 1, ships in the ONNX graph, and runs in the browser with the flex features
extracted live from the input text.

## Non-goals

- Changing `EmojiHead`, `StyleHead`, the encoder, or the GAN. FusionHead consumes
  a **detached** copy of `EmojiHead`'s logits; the raw head and encoder train
  exactly as today.
- Replacing the `overlap` retrieval method or `flexrank.ts`. The ranker and its
  output schema are unchanged.
- A separate training stage. FusionHead co-trains inside `LitEncoder`.

## Overview of the pipeline

```
text ──► TextEncoder ──► EmojiHead ──► emoji_logits ─┐(detached)
                                                     ├─► FusionHead ──► fusion_logits
CLDR ranker (flexrank) ──► per-emoji raw flex rows ──┘   (residual on emoji_logits)
                        ──► flexq row features ──────────┘
```

Three implementations of the ranker exist and must stay in parity:

| impl | consumer | source data |
|---|---|---|
| `tools/data/flexrank.ts` | writes `flexsearch`/`flexq` into train/eval jsonl (training signal) | `loadCldrAnnotations()` + `data/labels.json` |
| `model/flexrank.py` (new) | `tools/report.py` probe charts, `model/pred.py` | `web/public/flex.json` |
| `web/src/flexrank.js` (new) | browser inference | `web/public/flex.json` |

`web/public/flex.json` (new, written by `regen`) is the shared, pruned index the
Python and JS rankers read: for each of the ~950 vocab emoji, its lowercased CLDR
keyword list, plus a `{keyword: idf}` map (idf computed over the full CLDR set at
regen time). The TS ranker keeps using `loadCldrAnnotations()` directly; because
`flex.json` is derived from the same call in the same `regen` run, the three
rankers produce identical numbers. A shared fixture
(`web/src/flexrank.fixture.json`: `text -> {flexsearch, flexq}`) is asserted by
both `web/src/flexrank.test.js` and `model/test_flexrank.py`.

## Per-emoji raw flex vector (model input, 10 floats)

`data.py` / `flexrank.py` / `flexrank.js` all produce, per retrieved emoji, in
the flex list's sorted order:

```
[score, score_norm, exact, fuzzy, best_idf, rank_recip, n_kw, kw_len, word_len, overlap]
```

- `score, score_norm, exact, fuzzy, best_idf, n_kw, kw_len, word_len, overlap` —
  exactly the `flexsearch` array fields (`flexrank.ts:FLEX_COLS` minus `emoji`).
- `rank_recip = 1 / (position_in_list + 1)` — derived from array order (order is
  preserved in the jsonl and in the sparse store).

Emoji **not** in the list get an all-zero row; `present = (score > 0)` is derived
inside `FusionHead`, not stored.

`flexq` is a 5-vector row feature, broadcast to every emoji:
`[tokens, matched, sum, max, cand]`.

## FusionHead module (`model/model.py`)

```
class FusionHead(nn.Module):
    # 4 learned embeddings for the clamped int features
    self.emb_nkw   = nn.Embedding(FUSION_INT_CLAMP + 1, FUSION_INT_EMBED_SIZE)
    self.emb_kwlen = nn.Embedding(FUSION_INT_CLAMP + 1, FUSION_INT_EMBED_SIZE)
    self.emb_wlen  = nn.Embedding(FUSION_INT_CLAMP + 1, FUSION_INT_EMBED_SIZE)
    self.emb_ovl   = nn.Embedding(FUSION_INT_CLAMP + 1, FUSION_INT_EMBED_SIZE)
    self.net = nn.Sequential(
        nn.Linear(F_SCALAR + 4 * FUSION_INT_EMBED_SIZE, FUSION_HIDDEN, bias=True),
        nn.LeakyReLU(RELU_SLOPE),
        nn.Dropout(DROPOUT_FUSION),
        nn.Linear(FUSION_HIDDEN, 1, bias=True),
    )
    # final Linear zero-init  -> g(feat) starts at 0  -> fusion == EmojiHead at init

    def forward(self, emoji_logit, flex_raw, flexq):
        # emoji_logit: [B, V]   flex_raw: [B, V, 10]   flexq: [B, 5]
        scal, idx = fusion_features(emoji_logit, flex_raw, flexq)  # [B,V,18], [B,V,4] long
        emb = torch.cat([
            self.emb_nkw(idx[..., 0]), self.emb_kwlen(idx[..., 1]),
            self.emb_wlen(idx[..., 2]), self.emb_ovl(idx[..., 3]),
        ], dim=-1)                                                 # [B, V, 32]
        g = self.net(torch.cat([scal, emb], dim=-1))              # [B, V, 1]
        return emoji_logit + g.squeeze(-1)
```

`fusion_features(...)` is a module-level pure-torch function (traced into ONNX
unchanged, so **one** implementation covers train + export). It returns
`(scalars [B, V, 18], int_idx [B, V, 4] long)`:

- `present = (score > 0).float()`
- scalar features (`F_SCALAR = 18`):
  `emoji_logit`, `present`, `log1p(score)`, `score_norm`,
  `score / (flexq.sum + eps)`, `exact`, `fuzzy`, `exact / (flexq.tokens + eps)`,
  `fuzzy / (flexq.tokens + eps)`, `best_idf / 10`, `rank_recip`,
  `overlap / (word_len + eps)` (coverage — the one int-derived scalar we keep,
  since two independent embeddings can't express a ratio),
  and row-broadcast `flexq.tokens`, `flexq.matched`,
  `flexq.matched / (flexq.tokens + eps)`, `log1p(flexq.sum)`, `flexq.max`,
  `log1p(flexq.cand)`.
- `int_idx` = `clamp(round([n_kw, kw_len, word_len, overlap]), 0, FUSION_INT_CLAMP).long()`,
  gathered through the four embeddings in `forward`.

`fused_logit = emoji_logit + g`. Zero-init on the last Linear means an untrained
FusionHead reproduces `EmojiHead` and only departs as it learns.

## Config additions (`model/config.py`)

```
FUSION_HIDDEN         = 48
DROPOUT_FUSION        = 0.1
FUSION_INT_CLAMP      = 16
FUSION_INT_EMBED_SIZE = 8
```

Add them to the `CONFIG_NAME` hyperparameter string. No removals.

## Training integration (`model/train.py`)

- `ALL_HEADS = ("style", "emoji", "critic", "fusion")`. `fusion` requires `emoji`
  in the selected set (`_parse_heads` raises `typer.BadParameter` otherwise).
- `LitEncoder.__init__`: `if "fusion" in self.heads: self.fusion = FusionHead()`.
- `LitEncoder._step`: after `emoji_logits` is computed,
  ```
  if "fusion" in self.heads:
      fl = self.fusion(emoji_logits.detach(), flex_raw, flexq)
      loss_fusion = lse_infonce(fl, emoji, INFONCE_TEMP)
      loss = loss + loss_fusion
      self._log(f"loss/fusion/{split}", loss_fusion, bs)
  ```
  The batch tuple grows to `(text, emoji, style, colors, flex_idx, flex_raw, flexq)`
  (see Data loading). `emoji_logits.detach()` keeps the encoder + EmojiHead
  gradients identical to today.
- **Three MRRs**, logged per split over the `has_e` rows (same mask as `MRR/e`):
  - `MRR/e/{split}` — raw `emoji_logits` (unchanged).
  - `MRR/flex/{split}` — **new**. `flex_score_logits` = dense `score` scattered
    over the vocab, `-inf` where absent; `mrr(flex_score_logits, emoji)`. A fixed
    reference line (same every epoch).
  - `MRR/fusion/{split}` — **new**. `mrr(fusion_logits, emoji)`.
- `configure_optimizers`: `fusion` params already included by the existing
  `for h in self.heads: params += getattr(self, h).parameters()` loop.
- `_train_encoder` monitor / early-stop:
  ```
  "MRR/fusion/val" if "fusion" in heads
  else "F1/val" if {"emoji", "critic"} <= set(heads)
  else "MRR/e/val" if "emoji" in heads
  else "MRR/s/val" if "style" in heads
  else "auc/critic/val"
  ```
  `mode="max"` throughout. `save_pt(mod.fusion.state_dict(), out_dir/"fusion.pt", stage="enc")`.
- No-positional run: `ALL_HEADS` now includes `fusion`, so the default pipeline
  trains it and the export bakes it in.
- GAN stage: `_require_pt(pt_dir, [... , "fusion.pt"])`; `_run_remote` gan branch
  uploads `fusion_bytes`; `train_remote` gains a `fusion_bytes` param and writes
  it to `FUSION_PT`. `CODE_FILES` unchanged (training reads flex from the jsonl,
  which is already listed); `flex.json` is **not** needed on Modal (report is
  skipped there, export bakes only weights).

## Data loading (`model/data.py`)

- `read()` gains an optional capture of `flexsearch` (list of arrays) and `flexq`
  (dict); missing ⇒ empty list / zeros. `record` grows two fields.
- Dense `[n_rows, V, 10]` is ~10 GB — store **sparse** per row:
  - `flex_idx`  : `long[FLEX_MAX_K]` — vocab indices of retrieved emoji, `-1` pad
    (`FLEX_MAX_K = 32`, matches `--flex-k` default; rows are already capped).
  - `flex_raw`  : `float[FLEX_MAX_K, 10]` — the raw vectors incl. `rank_recip`
    from list position, zero pad.
  - `flexq`     : `float[5]`.
  Emoji whose glyph isn't in `emoji2idx` are dropped from the row's flex list
  before indexing.
- `EmojiDataset` stacks those; `__getitem__` returns
  `(text, emoji, style, colors, flex_idx, flex_raw, flexq)`.
- New `scatter_flex(flex_idx, flex_raw) -> [B, V, 10]` helper (zeros, then
  `scatter_` on dim 1 with the pad mask) used by `_step`, `report.py`, `pred.py`.

## Export & web

### `model/export_onnx.py`

- `_load(FusionHead(), FUSION_PT)`; abort if `fusion.emb_nkw`/etc. imply a vocab
  mismatch (reuse the existing `len(EMOJIS)` guard pattern via the last Linear /
  the fact that fusion output width is `V`).
- `ExportWrapper.forward(x, flex, flex_q)`:
  ```
  emb = self.enc(x)
  style_logits = self.style(emb)
  emoji_logits = self.emoji(emb)
  fusion_logits = self.fusion(emoji_logits, flex, flex_q)
  color = ...
  return style_logits, emoji_logits, fusion_logits, color
  ```
  Second input `flex`: shape `[batch, V, 10]` float32 (the raw per-emoji vectors,
  absent emoji zero). Third input `flex_q`: `[batch, 5]` float32 (row-level, not
  per-emoji). `dynamic_axes` adds `batch` for `flex`, `flex_q`, and
  `fusion_logits`.
- Output names: `["style_logits", "emoji_logits", "fusion_logits", "color"]`.
- `meta.json` gains `"flex_cols"` (the 10-name list) and `"flex_k"` so JS knows
  the input width / cap.

### `web/public/flex.json` (written by `regen`)

```
{ "emojis": ["😀", ...],                      # vocab order, == labels.json emojis
  "keywords": [["grin","face",...], ...],     # per-emoji lowercased CLDR keywords
  "idf": { "grin": 7.1, "face": 3.2, ... } }  # over the full CLDR set
```

`files.ts` / `files.py` gain `FLEX_JSON = "web/public/flex.json"`. `regen`
computes it from the same `loadCldrAnnotations()` + vocab it already has; written
next to the other `web/public/` artifacts and committed. Add a one-line note to
the `regen` summary.

### `web/src/flexrank.js` (new)

Port of `queryTokens`, `fuzzyMatch`, and the inverted-index scorer, reading the
parsed `flex.json`. Builds, per input text:

- `flexRaw`: `Float32Array(V * 10)` — the raw per-emoji vectors (same layout as
  the ONNX `flex` input), zeros for absent emoji.
- `flexQ`: `Float32Array(5)`.

Deterministic, byte-parity with `flexrank.ts` (locked by
`web/src/flexrank.fixture.json`). Perf: a few tokens × small posting lists +
one `V*10` zero-fill + ~200 scatter writes → well under 1 ms per call.

### `web/src/model.js` + inference caller

- Load `flex.json` alongside `meta.json` (one extra `fetch`, ~25 KB gzipped).
- Build `flexRaw` / `flexQ` from the normalized text, pass as ONNX inputs
  `flex` / `flex_q`. Read `style_logits`, `emoji_logits`, `fusion_logits`,
  `color`.
- **Emoji ranking mode** — a 3-way toggle in the UI, state in React and
  persisted to `localStorage`:
  1. **Fusion** (default) — rank by `sigmoid(fusion_logits)`.
  2. **Model** — rank by `sigmoid(emoji_logits)` (raw `EmojiHead`).
  3. **Keywords** — rank by the JS `FlexRanker` score directly
     (`flexRaw[v*10 + 0]` per vocab emoji, desc; emoji absent from the flex list
     sort last). No ONNX output needed — score is already computed client-side.
  The mode only changes which score orders / thresholds the emoji list; style
  and colour are unaffected. Default is **Fusion**.
- `web/src/flexrank.test.js` (new) — fixture parity + a couple of hand cases.

## Report (`tools/report.py`)

- New `model/flexrank.py` — `FlexRanker(flex_json_path)` with `.rank(text)`
  returning the sorted list of `(emoji, *10-vector)` and `.flexq(text)`; mirrors
  `flexrank.ts`. `model/test_flexrank.py` asserts the shared fixture.
- `_acc_at_k` reused. For a set of texts + targets, compute three curves:
  - `emoji` — `head(enc(texts))` (as today),
  - `flex` — dense `score` logits from `FlexRanker`, `-inf` absent,
  - `fusion` — `FusionHead(emoji_logits, scatter_flex(...), flexq)`.
- Charts that get all three lines (raw = solid `lline`, `flex` = `lline2`,
  `fusion` = `lline3`; legend `EmojiHead / FlexRank / Fusion`):
  - `_section_emoji` eval chart (keeps the dashed CLDR `baseline` too — legend
    gains a 4th dashed entry; small tweak to `_linechart`'s legend branch),
  - `_section_emoji` `keywords.json` probe chart,
  - `_section_cldr` `cldr.jsonl` probe chart,
  - `_section_cards` gold-set **emoji** chart. The current cards chart plots
    emoji + style on one axis; split it — emoji becomes the 3-way chart, and
    `style_acc_at_k` moves to its own single-line chart directly below.
- **FlexRank smoke test**: `cldr.jsonl` rows are CLDR keywords, so FlexRank
  `acc@1` there should be ≈1. `_section_cldr` records `flex_acc_at_k`; the header
  banner (`_header_html`) flags an issue if `flex_acc_at_k[0] < 0.7`
  ("FlexRank/flex.json looks broken").
- `report.json` mirrors the new keys (`flex_acc_at_k`, `fusion_acc_at_k` under
  each section).
- `.lline3` CSS (a third distinct stroke colour, e.g. teal `#0a9c8b`).
- Requires `fusion.pt` + `web/public/flex.json`; missing ⇒ the fusion/flex curves
  are omitted and the banner notes it (same softness as missing `gen.pt`).

## `model/pred.py`

Load `fusion.pt` + `model/flexrank.py`; for each of the 200 rows also emit
`fusion_top_labels` (same threshold rule as `emoji`) so `data/pred.jsonl` shows
raw vs fused side by side.

## Files touched

| file | change |
|---|---|
| `model/config.py` | 4 `FUSION_*` consts; `CONFIG_NAME` |
| `model/model.py` | `FusionHead`, `features()` helper |
| `model/data.py` | `read()` captures flex; sparse `flex_idx`/`flex_raw`/`flexq` in `EmojiDataset`; `scatter_flex()` |
| `model/train.py` | `fusion` in `ALL_HEADS`; `_step` fusion loss + 3 MRRs; monitor → `MRR/fusion/val`; save `fusion.pt`; gan `_require_pt`/upload plumbing |
| `model/export_onnx.py` | load `fusion.pt`; 2 new inputs, `fusion_logits` output; `meta.json` `flex_cols`/`flex_k` |
| `model/flexrank.py` | **new** — Python ranker over `flex.json` |
| `model/pred.py` | fused predictions |
| `model/runmeta.py` | — (uses existing `save_pt`/`load_pt`) |
| `files.py` / `files.ts` | `FLEX_JSON`, `FUSION_PT` |
| `tools/data/flexrank.ts` | export a `buildFlexJson(vocab)` used by `regen` |
| `tools/data/regen.ts` | write `web/public/flex.json`; summary line |
| `web/src/flexrank.js` + `web/src/flexrank.test.js` | **new** |
| `web/src/model.js` + inference caller | load `flex.json`, build inputs, read `fusion_logits` |
| `web/src/flexrank.fixture.json` | **new** — shared parity fixture |
| `tools/report.py` | `model/flexrank.py` use; 3-way charts; `.lline3`; smoke-test banner |
| `tools/test_report.py`, `model/test_train_cli.py`, `model/test_flexrank.py` | updated / new |
| `CLAUDE.md` | FusionHead, `fusion` head, `flex.json`, the 3 parity surfaces |

## Testing / verification

- `uv run ruff check . && uv run ruff format --check .`
- `uv run python model/test_runmeta.py`, `model/test_train_cli.py`,
  `model/test_flexrank.py`, `tools/test_report.py`
- `cd web && npm test` (adds `flexrank.test.js`)
- `bun run regen` — writes `web/public/flex.json`; confirm parity fixture.
- `train enc --local --heads emoji,fusion` — watch `MRR/e/val`,
  `MRR/flex/val`, `MRR/fusion/val`; `MRR/fusion/val ≥ max(MRR/e/val,
  MRR/flex/val)` is the success signal. Then full `train --local`.
- `uv run python model/export_onnx.py` then `cd web && npm run build`; load the
  app, confirm emoji predictions shift vs a raw-head build and inference stays
  interactive.
- `uv run python tools/report.py` — eval / keyword / cldr / cards charts show
  three lines; cldr FlexRank `acc@1` ≈ 1.

## Rollout

`flex.json` and the new `.pt` don't exist yet: a fresh `bun run regen` (for
`flex.json`) followed by a full retrain + `export_onnx.py` is required before the
web build. Until then `report.py` / the web omit the fusion curves rather than
error.

## Risks / open questions

- **Three ranker implementations.** Mitigated by the shared `flex.json` + the
  shared fixture asserted from both JS and Python; the TS ranker is covered by
  its own `cldr-baseline` parity check plus `flex.json` being derived from the
  same `loadCldrAnnotations()` run.
- **GPU memory**: the `[512, V, ~50]` fusion-feature tensor is ~100 MB plus a
  similar hidden activation — fine on a T4 next to the rest, but worth watching
  the first GPU run; `FUSION_HIDDEN` / GPU batch size are the levers.
- **`MRR/fusion/val` as the sole monitor** means a regression in the raw
  `EmojiHead` (should be impossible given the detach, but e.g. a shared-seed
  interaction) wouldn't be caught by early stop — `MRR/e/val` stays logged for
  eyeballing.
- **`n_kw` for absent emoji is 0** (embedding row 0 = "not retrieved"), so the
  head can't use "EmojiHead likes a 40-keyword attractor that flex didn't
  retrieve" as a demotion signal in v1. Easy v2 extension (fill true `n_kw` for
  all vocab from `flex.json`).
