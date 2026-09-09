# KW-Fusion rework — design

## Summary

Replace the current `FusionHead` (a per-label residual re-ranker over 18+4
hand-crafted features derived from a per-emoji CLDR-retrieval ranking) with a
three-head design that blends a **semantic** and a **lexical** query vector in
the shared emoji-embedding space:

```
q_txt   = EmojiHead(text_embedding)          # 64-d
q_kw    = KWHead(tf_vec)                      # 64-d
a       = sigmoid(FusionHead([text_embedding, tf_vec]))   # per-row scalar
q_fused = a * q_txt + (1 - a) * q_kw          # 64-d
fusion_logits = q_fused @ E.t() + bias        # E, bias shared with EmojiHead
```

`tf_vec` is a length-`N` (N dynamic, ~876) soft term-frequency vector over a fixed
global keyword vocabulary, produced by the same three ranker surfaces that
exist today (`tools/data/flexrank.ts`, `model/flexrank.py`,
`web/src/flexrank.js`), kept in parity by the conformance fixture.

## Motivation

- `MRR/fusion/val` does not clear `max(MRR/e/val, MRR/flex/val)` by a
  meaningful margin — the residual-on-features design is too weak.
- The 18+4 engineered feature block (`fusion_features`, `FUSION_INT_*`) is
  brittle and hard to reason about.
- The lexical signal is collapsed into emoji space by `flexrank` and confined
  to the top-32 emojis it surfaces before the head sees it. A learned
  `KWHead: tf_vec -> 64-d` can light up any emoji in the vocab from keyword
  evidence.

This is a clean replacement (chosen over coexistence): the entire per-emoji
flex plumbing is removed, not kept alongside.

## Current design (removed by this rework)

- `model/model.py`: `fusion_features`, `FusionHead` (the MLP residual), the
  `emb_nkw`/`emb_kwlen`/`emb_wlen`/`emb_ovl` int-embeddings.
- `model/data.py`: `_row_flex`, `scatter_flex`, `FLEX_MAX_K`, `FLEX_RAW_DIM`,
  `FLEXQ_DIM`, `_FLEXQ_KEYS`, and the `flex_idx` / `flex_raw` / `flexq`
  dataset tensors.
- `regen` per-row fields `flexsearch` (32x10 array) and `flexq` (5 stats).
- `flex.json` keys `emojis`, `keywords`, `idf`, `idf_default`.
- `model/flexrank.py:FlexRanker.rank` / `.flexq` (per-emoji ranking).
- `tools/report.py`: the raw-FlexRank `acc@k` overlay curve, the CLDR-section
  FlexRank smoke curve, and `FLEX_SMOKE_MIN`.
- `model/config.py`: `FUSION_HIDDEN`, `DROPOUT_FUSION`, `FUSION_INT_CLAMP`,
  `FUSION_INT_EMBED_SIZE`.
- `meta.json`: `flex_cols`, `flex_k`.

## Target architecture (`model/model.py`)

`V = len(EMOJIS)` (957 today, dynamic). `N` = keyword vocab size (dynamic, ~876).

### `EmojiEmbedding` (new, shared)

Owns what `EmojiHead` used to own:

```
self.embed = nn.Embedding(V, EMOJI_EMBED_SIZE)   # E
self.bias  = nn.Parameter(torch.zeros(V))
def score(self, q):   # q: [B, 64]
    return q @ self.embed.weight.t() + self.bias
```

Instantiated once by `LitEncoder` whenever `emoji` is in `--heads`. Passed by
reference to `EmojiHead` and `KWHead`. Serialized as `emoji_embed.pt`.

### `EmojiHead` (unchanged computation)

```
self.net = nn.Sequential(
    nn.Dropout(p=DROPOUT_EMOJI),
    nn.Linear(TEXT_EMBED_SIZE, EMOJI_EMBED_SIZE, bias=False))
def forward(self, text_embedding):
    return self.net(text_embedding)          # q_txt, [B, 64] — no scoring here
```

Standalone logits (for `MRR/e/val`, pred, export, report) =
`emoji_embedding.score(q_txt)`. Serialized as `emoji.pt`.

### `KWHead` (new, structurally identical)

```
self.net = nn.Sequential(
    nn.Dropout(p=DROPOUT_KW),
    nn.Linear(N, EMOJI_EMBED_SIZE, bias=False))   # Linear weight zero-init
def forward(self, tf_vec):
    return self.net(tf_vec)                  # q_kw, [B, 64]
```

`N` is read at construction from `data.py` (`FLEX_N`, = `len` of `flex.json` `kw_vocab`). Zero-init on the `Linear` weight so `q_kw = 0`
at step 0. Standalone logits (for `MRR/kw/val`, the web `keywords` toggle) =
`emoji_embedding.score(q_kw)`. Serialized as `kw.pt`.

### `FusionHead` (now just the gate)

```
self.net = nn.Linear(TEXT_EMBED_SIZE + N, 1)   # weight zero-init, bias +4
def forward(self, text_embedding, tf_vec):
    return torch.sigmoid(
        self.net(torch.cat([text_embedding, tf_vec], dim=-1))).squeeze(-1)  # a, [B]
```

No hidden layer. Weight zero-init + bias `+4` => `a ~= 0.982` for every row at
step 0, so with `q_kw = 0` the fused logits start at `0.982 * (q_txt @ E.t()) +
bias` — within a hair of `EmojiHead`, no day-one regression. Serialized as
`fusion.pt`. `FUSION_HIDDEN` / `DROPOUT_FUSION` are gone.

### Fused output

```
q_fused = a.unsqueeze(-1) * q_txt.detach() + (1 - a).unsqueeze(-1) * q_kw.detach()
fusion_logits = emoji_embedding.score(q_fused)
```

### Gradient flow (coupling)

- `q_txt` and `q_kw` are **detached** in the fused path: `loss/fusion` trains
  **only the gate** (`FusionHead.net`).
- `E` / `bias` take gradient from `loss/emoji` and `loss/kw` both — the shared
  table is genuinely co-trained by the semantic and lexical objectives.
- The trunk and `EmojiHead.net` see gradient only from `loss/emoji` (and the
  other selected heads), never from `loss/fusion`.
- Consequence: with `fusion` selected, `E`/`bias` are additionally shaped by
  `loss/kw`, so `MRR/e/val` shifts slightly vs. an `emoji`-only run. The
  trunk and `EmojiHead.net` are unaffected by fusion. `MRR/e/val` and
  `MRR/kw/val` remain each head's standalone ceiling; `MRR/fusion/val` is the
  best convex blend of the two.

## Lexical TF vector

### Keyword vocabulary (`kw_vocab`, built by `regen`, TS side)

`buildFlexRanker(corpusTexts: string[])` — `regen` passes the `data/data.jsonl`
master texts (**not** `data/cldr.jsonl`, whose row texts are the keywords
themselves and would make the corpus filter a no-op).

1. Collect every distinct keyword string across all glyphs'
   `loadCldrAnnotations()` lists, `.trim().toLowerCase()`, deduped per glyph.
2. Compute `df[kw]` = number of distinct glyphs the keyword appears in.
3. Candidate filter — keep `kw` iff **all** of:
   - `df[kw] === 1` (max IDF — appears in exactly one emoji's keyword set),
   - `queryTokens(kw)` yields exactly `[kw]` (single `[a-z0-9]+` token, >= 2
     chars, non-stopword),
   - `KW_MIN_LEN <= kw.length <= KW_MAX_LEN` (6..10),
   - `kw` occurs >= 1x as a `queryTokens` token across `corpusTexts`.
4. Sort the survivors alphabetically. This ordered list is `kw_vocab`.
   **Use all of them** — no fixed size, no truncation.

`N = kw_vocab.length` is therefore **dynamic** (~876 on the current
`data/data.jsonl`) and drifts as the master grows. Nothing downstream hardcodes
it: `FLEX_N` is read from `flex.json` / `meta.json`.

IDF here is just `df === 1`; `makeIdf` is not needed. `df` and the corpus scan
are **build-time-only** — not needed at serve time, dropped from `flex.json`.

New constants in `tools/data/config.ts`: `KW_MIN_LEN = 6`, `KW_MAX_LEN = 10`,
`MIN_FUZZY_SCORE = 0.66`. `FUZZY_MIN_LEN` (4) is reused from
`tools/analysis/cldr-baseline.ts`. (No `KW_VOCAB_SIZE`.)

`tools/analysis/cldr-keywords.ts` (standalone) prints the max-IDF keyword-count
matrix over min-length 5..8 x max-length 8..11 — the tool used to pick the
`[6, 10]` band.

### `tf_vec(text)` — the one shared surface

For each keyword `k` in `kw_vocab` (index `j` = position in the list):

```
tf[j] = sum over w in query_tokens(text) of overlap(w, k)

overlap(w, k):
    if w == k:                                  return 1.0
    if len(w) < FUZZY_MIN_LEN or len(k) < FUZZY_MIN_LEN:   return 0.0
    if not (w.startswith(k) or k.startswith(w)):           return 0.0
    r = min(len(w), len(k)) / max(len(w), len(k))
    return r if r >= MIN_FUZZY_SCORE else 0.0
```

- Raw overlap sum. No IDF weighting on the cell values, no normalization
  (mirrors `EmojiHead` consuming the raw, un-normalized text embedding).
- Fuzzy contributions land in `[0.66, 1.0)`, graded by length ratio; anything
  weaker is dropped. `FUZZY_WEIGHT` (0.6) and `FUZZY_MAX_LEN_DELTA` (3) are
  **not** used on this path (the 0.66 cutoff is a stricter relative bound).
- Brute-force: compare each query token to all `N` (~876) keywords. No inverted
  index.
- `query_tokens` is the existing shared tokenizer (`cldr-baseline.ts` /
  `flexrank.py` / `flexrank.js`). Cell values `r3`-rounded (3 decimals) to keep
  the three surfaces bit-identical.
- This rule is **local to `tf_vec`**. `tools/analysis/cldr-baseline.ts`'s
  `overlap` (behind the report's CLDR-baseline `acc@k` overlay) is unchanged —
  it stays a fixed external reference.

### `flex.json` / `meta.json`

- `web/public/flex.json` becomes `{ "kw_vocab": string[N] }` (N ~876).
- `meta.json`: remove `flex_cols`, `flex_k`; add `flex_kw` (the `kw_vocab`
  names, for parity/debug) and `flex_n` (= N).

### Per-row field (`regen` output)

`flexsearch` + `flexq` are replaced by one **`flex_tf`** field, stored sparse
as a pair array of `[keyword_index, value]` (a query hits 0..a-handful of
keywords):

```json
"flex_tf": [[12, 1.0], [88, 0.667]]
```

`regen` flags: `--flex-k 32` removed (no size knob); `--no-flexsearch` -> `--no-kw`
(skips the `flex_tf` field, `flex.json`, and the fixture).

### `model/data.py`

- New module constants: `KW_VOCAB` / `FLEX_N` (read from `flex.json` `kw_vocab`).
- `read` captures `flex_tf` onto each `record` (replaces the `flexsearch` /
  `flexq` capture).
- New `_row_tf(row) -> torch.Tensor [FLEX_N]`: zeros, then scatter the sparse
  pairs. Replaces `_row_flex`.
- Dataset holds one `self.flex_tf` tensor `[num_rows, FLEX_N]`; `__getitem__`
  returns `flex_tf` in place of `flex_idx` / `flex_raw` / `flexq`.

### Conformance fixture

`web/src/flexrank.fixture.json` locks the `tf_vec` output (dense length-N,
`r3`-rounded) for the 8 `FLEX_FIXTURE_TEXTS`, regenerated by `regen` in the
same pass, replayed by `model/test_flexrank.py` and `web/src/flexrank.test.js`.
This is now the **sole** cross-surface drift guard.

## Training loop (`model/train.py`, `LitEncoder`)

- `--heads` unchanged: `style,emoji,critic,fusion`; `fusion` still requires
  `emoji`. `fusion` selected -> instantiate `EmojiEmbedding` + `EmojiHead` +
  `KWHead` + `FusionHead`. `emoji` without `fusion` -> `EmojiEmbedding` +
  `EmojiHead` only.
- Batch tuple: `text, emoji, style, colors, flex_idx, flex_raw, flexq`
  becomes `text, emoji, style, colors, flex_tf` (one `[B, N]` tensor).
- Losses (all `lse_infonce` at `INFONCE_TEMP`), summed into `loss`:
  - `loss/emoji` on `emoji_embedding.score(q_txt)` — as today.
  - `loss/kw` on `emoji_embedding.score(q_kw)` — only when `fusion` selected.
  - `loss/fusion` on `emoji_embedding.score(q_fused)` — `q_txt` / `q_kw`
    detached, so only `FusionHead.net` is trained by it.
- Logged: `MRR/e/{train,val}`, `MRR/kw/{train,val}`, `MRR/fusion/{train,val}`.
  `MRR/flex/val` is removed (was the per-emoji raw ranker MRR); `MRR/kw/val`
  takes its place.
- Checkpoint / early-stop key order: `MRR/fusion/val` (fusion on) -> `F1/val`
  (emoji + critic) -> `MRR/e/val` -> `MRR/s/val` -> `auc/critic/val`.
  `MRR/fusion/val` is still expected `>= max(MRR/e/val, MRR/kw/val)`.
- Writes `enc.pt`, `style.pt`, `critic.pt`, `emoji.pt`, `emoji_embed.pt`, and
  (fusion on) `kw.pt`, `fusion.pt`.

### `gan` stage

Frozen-encoder path unchanged. Required-files check adds `emoji_embed.pt` and
`kw.pt`: `enc.pt`, `critic.pt`, `style.pt`, `emoji.pt`, `emoji_embed.pt`,
`kw.pt`, `fusion.pt`.

## Checkpoints & path constants

| file | module |
|---|---|
| `emoji_embed.pt` | `EmojiEmbedding` (shared table `E` + `bias`) |
| `emoji.pt` | `EmojiHead` (`Dropout -> Linear(620, 64)`) |
| `kw.pt` | `KWHead` (`Dropout -> Linear(N, 64)`) |
| `fusion.pt` | `FusionHead` (`Linear(820, 1)`) |

- `files.py` / `files.ts`: add `EMOJI_EMBED_PT`, `KW_PT` (keep `FUSION_PT`).
- The label-count-vs-`labels.json` abort check (in `export_onnx.py`, and the
  `report.py` provenance banner) moves from `emoji.pt` to `emoji_embed.pt`
  (the `[V, 64]` table is where `V` lives).
- `runmeta` provenance (train_sha / commit consistency) covers
  `emoji_embed.pt` and `kw.pt` alongside the rest.

## ONNX export (`model/export_onnx.py`)

- Loads `enc.pt`, `style.pt`, `emoji.pt`, `emoji_embed.pt`, `kw.pt`,
  `fusion.pt`, `gen.pt`.
- `ExportWrapper` inputs: `input` (int64 char ids) + `flex_tf` `[1, N]`
  (replaces `flex` `[1, V, 10]` and `flex_q` `[1, 5]`).
- Outputs: `style_logits`, `emoji_logits` (`score(q_txt)`), `kw_logits`
  (`score(q_kw)`, new), `fusion_logits` (`score(q_fused)`), `color` (`[5, 9]`,
  unchanged).
- `meta.json`: as in the flex.json section (`flex_kw`, `flex_n`; no
  `flex_cols` / `flex_k`).
- Abort if `emoji_embed.pt` / `style.pt` label counts disagree with
  `data/labels.json`.

## Web (`web/src/`)

- `flexrank.js:makeFlexRanker(flex.json)` returns only the dense length-N
  `flex_tf` vector for a text; no ranking, no `flexq`. `tf_vec` byte-matches
  the TS / Python surfaces (fixture-locked).
- `model.js` passes `flex_tf` as the single lexical model input (drops the
  `flex` / `flex_q` construction).
- Masthead 3-way toggle maps straight to logit outputs: `fusion` ->
  `fusion_logits`, `model` -> `emoji_logits`, `keywords` -> `kw_logits`.
  `localStorage` key `emojiMode` unchanged.
- `decodeColors` / `decodeColorList` unchanged.

## Report (`tools/report.py`)

- Emojis section: the 3-way `acc@k` overlay becomes **EmojiHead / KWHead /
  Fusion** (raw-FlexRank curve removed). KWHead curve drawn when `kw.pt` +
  `fusion.pt` load and rows carry `flex_tf`.
- CLDR section: the FlexRank smoke curve and `FLEX_SMOKE_MIN` provenance check
  are removed (the conformance fixture is the drift guard now). The CLDR
  `acc@k` probe over `data/cldr.jsonl` and the CLDR-baseline overlay stay.
- Cards section: unchanged (enc + style + emoji + gen). `report.json` mirrors
  the new curve set.
- Provenance banner: `.pt` set is `enc` / `emoji_embed` / `emoji` / `kw` /
  `fusion` / `style` / `gen`.

## Config (`model/config.py`)

- Add `DROPOUT_KW`.
- Remove `FUSION_HIDDEN`, `DROPOUT_FUSION`, `FUSION_INT_CLAMP`,
  `FUSION_INT_EMBED_SIZE`.
- TF-vector knobs (`KW_MIN_LEN`, `KW_MAX_LEN`,
  `MIN_FUZZY_SCORE`) live on the TS build side only; Python reads `flex_n`.

## Migration

1. `bun run regen` — rebuilds `train.jsonl` / `eval.jsonl` / `labels.json`
   with the `flex_tf` field, writes the new `flex.json`, regenerates
   `flexrank.fixture.json`.
2. `.pt` files are gitignored: retrain from scratch with `train --local`. No
   checkpoint migration.
3. `uv run python model/export_onnx.py`; commit the regenerated
   `web/public/{model.onnx,meta.json,config.json,flex.json}` and
   `web/src/flexrank.fixture.json`.
4. Update `CLAUDE.md` (fusion paragraph, `train.py` bullet, `config.py` bullet,
   `flex.json` / web bullets) and `docs/model.md` via the `update-model-md`
   skill.

## Verification

- `uv run ruff check .` && `uv run ruff format --check .`
- `uv run python model/test_runmeta.py`
- `uv run python model/test_train_cli.py`
- `uv run python model/test_flexrank.py`
- `uv run python tools/test_report.py`
- `cd web && npm test` (`flexrank.test.js`, `feelings.test.js`,
  `model.test.js`)
- Real `train --local`: watch `MRR/fusion/val` vs. `MRR/e/val` /
  `MRR/kw/val` in TensorBoard (expect `MRR/fusion/val >= max(...)`), plus
  `energy/gan/val` vs. `energy/gan/ref` for stage 2.
- `uv run python model/pred.py --pt pt` — spot-check `data/pred.jsonl`
  (`fusion_top_labels` column present).
- `uv run python tools/report.py` — `report/<ts>-<sha>/report.html`, confirm
  the EmojiHead / KWHead / Fusion overlay renders and Cards section is intact.

## Non-goals

- No change to `TextEncoder`, `StyleHead`, `ColorGen`, `ColorCritic`, the GAN
  stage, or the energy metric.
- No coexistence path / no `fusion2` head — the old `FusionHead` internals and
  the per-emoji flex plumbing are deleted outright.
- `tools/analysis/cldr-baseline.ts` and the report's CLDR-baseline overlay are
  untouched.
- No new training stage; `KWHead` / `FusionHead` co-train inside `LitEncoder`.

## Risks / open questions

- **kw_vocab is all df==1 keywords** (each points at ~1 emoji), corpus-filtered
  to ones users actually type (len 6..10). ~876 today; ~68% appear >=1x in the
  corpus, ~20% of rows activate >=1 cell, so `KWHead` trains on a thin slice and
  the gate `a` sits near 1 for most rows (the designed fallback). If
  `MRR/kw/val` is poor, widen the length band or relax `df==1` — a `regen`-side
  change only, no model edits.
- **Linear gate** may be too weak to express "high TF but conflicting
  semantic confidence". First knob if `MRR/fusion/val` underperforms: one
  hidden layer in `FusionHead`.
- Detaching `q_txt` / `q_kw` in the fused path means the gate cannot reshape
  the representations to be more separable — deliberate, to keep the standalone
  MRRs interpretable. Revisit only if the blend plateaus below both inputs.
