# Simple fusion: three FusionHeads + offline uFuzzy keyword payload

## Problem

The current architecture to balance `EmojiHead` text prediction against keyword
prediction is heavy: a learned `KWHead` (`Linear(FLEX_N -> 64)`), a learned
`FusionHead` gate over `TEXT_EMBED_SIZE + FLEX_N` features, a report-driven 1000-word
`kw_vocab` selected by `tools/data/kwvocab.ts`, a soft-TF fuzzy matcher
(`_overlap`) reimplemented byte-identically in three languages
(`tools/data/flexrank.ts`, `model/flexrank.py`, `web/src/flexrank.js`) and locked by
a conformance fixture (`web/src/flexrank.fixture.json`), a second ONNX input
(`flex_tf`) and two extra ONNX outputs (`kw_logits`, `fusion_logits`), and a
`regen` step that is no longer deterministic because its `flex_tf` vocab depends on
the newest report.

## Goal

Replace it with a lightweight, maintainable, robust design that boosts held-out
emoji performance:

- `EmojiHead` trains **standalone** (`lse_infonce`, no keyword awareness) and stays
  usable as the sole predictor.
- The keyword predictor is a **non-learned** lookup: for each word of the
  `normalize`d text, fuzzy-match against `data/ii.json` keys with
  **uFuzzy** (`@leeoniya/uFuzzy`), project the matched keys onto the emoji vocab,
  aggregate to one score per emoji. uFuzzy runs **once offline in `regen.ts`** over
  every train/eval row and **live in the browser** on user input. There is exactly
  one uFuzzy implementation; Python never reimplements it. Word-splitting is
  `normalize(text).split(" ")` — the already-shared `normalize` (byte-identical
  across `model/data.py`, `tools/data/normalize.ts`, `web/src/model.js`) plus a
  space split; **no bespoke tokenizer**.
- A **minimal learned combiner** replaces `FusionHead`. Three variants are
  implemented and trained **in parallel in one run** so they can be compared:
  `FusionHeadGate`, `FusionHeadGain`, `FusionHeadMix`. Each consumes the same
  **detached** `(logit_m, kw)` inputs, so fusion loss can never regress the
  standalone predictor.
- Comparison surface is `tools/report.py` (a 5-way `acc@k` overlay on
  `data/eval.jsonl`). The browser ships **one** winner variant, picked at export
  time by a constant.

## Non-goals

- No Python port of uFuzzy. `report.py` / `pred.py` read the precomputed `kw`
  payload from the rows; they do not recompute keyword matches.
- No fused `acc@k` curves for the `data/keywords.json` / CLDR probes in the report
  (those probe strings carry no `kw` payload). First-cut limit, noted below.
- No change to `StyleHead`, `ColorCritic`, `ColorGen`, the GAN stage, `normalize`,
  the char vocab, or `INFONCE_TEMP`.

---

## 1. Keyword predictor

### 1.1 Projection (`regen.ts`, build time)

From `data/ii.json` (`{keyword: [emoji, ...]}`, ~5031 keys):

```
proj = { k: [emoji2idx[e] for e in ii[k] if e in EMOJIS] for k in ii }
proj = { k: v for k, v in proj if v }          # drop empty projections
```

`emoji2idx` indexes `data/labels.json` `emojis` (same order `meta.json` uses).
`proj` is vocab-dependent and recomputed on every `regen`. Because nothing about it
reads a report, **`regen` is deterministic again** for a fixed
`data/data.jsonl` + flags.

### 1.2 Word splitting — no bespoke tokenizer

The needle words are `normalize(text).split(" ")` filtered to `length >= 3`.
`normalize` is already shared and byte-identical across `model/data.py`,
`tools/data/normalize.ts`, and `web/src/model.js` (it lowercases, collapses
whitespace, trims, collapses 3+ char runs to 2, drops non-vocab chars). No
stopword list, no punctuation regex, no `tokenize.{ts,js}` file. uFuzzy does its
own intra-term matching; we only hand it one word at a time.

`model/tokenize.py` is still added — but **only** for the `report.py` "Keyword
vocab" diagnostic (`query_tokens(k) == [k]` single-token filter over
`data/ii.json` keys). It is Python-only, off the training/inference path, and
carries no cross-language parity burden.

### 1.3 uFuzzy index and query

Build one uFuzzy instance and search the keyword list `keys = Object.keys(proj)`
as the haystack:

```js
import uFuzzy from "@leeoniya/uFuzzy"
const uf = new uFuzzy({ intraIns: 1 })   // allow 1 extra needle char per term (plurals)
```

Per row / per user input: `words = normalize(text).split(" ").filter(w => w.length >= 3)`.
For each `word`:

```js
const idxs = uf.filter(keys, word)          // null when nothing matches
if (idxs && idxs.length) {
  const info = uf.info(idxs, keys, word)
  for (let i = 0; i < info.idx.length; i++) {
    const k = keys[info.idx[i]]
    const sim = info.chars[i] / k.length     // matched chars / keyword length, in (0, 1]
    if (sim >= 0.5) best.set(k, Math.max(best.get(k) ?? 0, sim))
  }
}
```

`sim` replaces Fuse's `1 - score`: 1.0 when the whole keyword is covered by the
word. `0.5` is the one tunable floor (drops weak partial matches). `intraIns: 1`
is the one tunable uFuzzy option.

### 1.4 Aggregate to per-emoji score

For emoji index `e`:

```
kw(e) = max over matched keys k with e in proj[k] of  sim(k) * w_k
w_k   = 1 / log2(1 + len(proj[k]))
```

`w_k` down-weights keys that fan out to many emoji (e.g. `"143"` -> 22 hearts).
`kw(e)` is `0` for every emoji not reached by a matched key, and `0` means **"no
opinion"**, never a negative vote.

### 1.5 Row field

`regen.ts` writes a sparse `kw` field on every train/eval row:

```
"kw": [[emojiIdx, value], ...]        # value > 0, rounded to 3 dp
```

Replaces `flex_tf`. `--no-kw` skips the field (and skips the web artifact in 1.6).
The `regen` summary line reports `kw` nonzero-count mean instead of the `flex_tf`
stats.

### 1.6 Web artifact

`regen.ts` writes `web/public/kwproj.json` (the same way it writes
`web/public/flex.json` today):

```json
{ "proj": { "<keyword>": [<emojiIdx>, ...], ... } }
```

`emojiIdx` in `data/labels.json` `emojis` order (== `meta.json` order). `w_k` is
recomputed in JS from `proj[k].length`. Replaces `web/public/flex.json`. Skipped
under `--no-kw`. `model/export_onnx.py` does **not** touch this file (same as
`flex.json` today — a bare `export_onnx` refresh leaves it as the last `regen`
wrote it).

---

## 2. The three FusionHeads (`model/model.py`)

Shared contract:

```
forward(logit_m: [B, N], kw: [B, N]) -> [B, N]      # N = len(EMOJIS)
```

`logit_m` is **already detached by the caller**. `z(x)` is the per-row standardize
over the emoji axis: `(x - x.mean(-1, keepdim)) / (x.std(-1, keepdim) + 1e-6)`.
Every variant is a no-op on emoji with `kw(e) == 0` at init and by construction of
the additive/interpolating form.

### `FusionHeadGain` (B)

- Parameter: `raw_beta` scalar, `nn.Parameter(torch.zeros(()))`.
- `fused = logit_m + softplus(raw_beta) * kw`
- Log `gain/beta` = `softplus(raw_beta)`.

### `FusionHeadMix` (C)

- Parameter: `raw_g` scalar, `nn.Parameter(torch.full((), -2.0))` -> `g ~= 0.12`,
  starts model-leaning.
- `g = sigmoid(raw_g)`
- `fused = (1 - g) * z(logit_m) + g * kw`
- Log `mix/g` = `g`.

### `FusionHeadGate` (A)

- Modules: `BatchNorm1d(6, affine=False)` then `Linear(6, 1)` with weight and bias
  zero-init -> `a = sigmoid(0) = 0.5` at start.
- Features per row, stacked to `[B, 6]`:
  1. `logit_m.max(-1)`
  2. top1 minus top2 of `logit_m`
  3. `entropy(softmax(logit_m, -1))`
  4. `kw.max(-1)`
  5. `(kw > 0).sum(-1)` as float
  6. `kw.sum(-1)`
- `a = sigmoid(linear(bn(feats)))`  -> `[B, 1]`
- `fused = a * z(logit_m) + (1 - a) * kw`
- Log `gate/a` = `a.mean()`.
- BN running stats (`running_mean[6]`, `running_var[6]`) are used at eval and
  exported to `meta.json` for the browser.

`KWHead` and the old `FusionHead` are deleted.

---

## 3. Training (`model/train.py`)

### 3.1 Heads

`--heads` surface unchanged: `style,emoji,critic,fusion`; `fusion` still requires
`emoji`. `fusion` in the head set now builds all three variants:

```python
self.fusion = nn.ModuleDict({
    "gate": FusionHeadGate(),
    "gain": FusionHeadGain(),
    "mix":  FusionHeadMix(),
})
```

`self.kw` (the `KWHead`) is removed. `configure_optimizers` already does
`params += list(getattr(self, h).parameters())` for each head name, so
`getattr(self, "fusion")` (the `ModuleDict`) contributes all three variants'
parameters; the explicit `self.kw.parameters()` line is deleted.

### 3.2 Step

In `_step`, unpack `text, emoji, style, colors, kw = batch` (the 5th element is now
the dense `kw` tensor, see section 4). After the `emoji` block has computed
`q_txt` and `emoji_logits`:

```python
if "fusion" in self.heads:
    lm = emoji_logits.detach()
    for name, head in self.fusion.items():
        fused = head(lm, kw)
        loss_v = lse_infonce(fused, emoji, INFONCE_TEMP)
        loss = loss + loss_v
        self._log(f"loss/fusion_{name}/{split}", loss_v, bs)
        if n_e:
            frr = mrr(fused[has_e], emoji[has_e]).mean()
        else:
            frr = torch.zeros((), device=emoji.device)
        self._log(f"MRR/fusion_{name}/{split}", frr, max(n_e, 1))
    # best-of-three under the existing key name
    self._log(f"MRR/fusion/{split}", <max of the three frr>, max(n_e, 1))
    self._log(f"gate/a/{split}", ...)   # from the gate variant
    self._log(f"gain/beta/{split}", ...)
    self._log(f"mix/g/{split}", ...)
```

`emoji_logits.detach()` means the fusion losses never touch the encoder,
`EmojiHead`, or `EmojiEmbedding`. The `emoji` head keeps its own
`loss_emoji`; with `fusion` selected, `loss_emoji` **is** added to the total (the
current code skips it when `fusion` is present — that changes: `EmojiHead` must be
trained). So: drop the `if "fusion" not in self.heads:` guard around
`loss = loss + loss_emoji`.

`MRR/kw/{split}` is still logged as a diagnostic: `mrr` of the raw `kw` tensor
(ranked descending) against `emoji`, for `has_e` rows.

### 3.3 Checkpoint / early-stop

The `monitor` ladder is unchanged: `MRR/fusion/val` (when `fusion` selected) ->
`F1/val` -> `MRR/e/val` -> `MRR/s/val` -> `auc/critic/val`, all `mode="max"`.
`MRR/fusion/val` is now the best of the three variants at each validation.

### 3.4 `.pt` outputs

`_train_encoder` writes, when `fusion` selected:

- `fusion_gate.pt`, `fusion_gain.pt`, `fusion_mix.pt` (each
  `save_pt(mod.fusion["<name>"].state_dict(), ..., stage="enc")`)

`kw.pt` and `fusion.pt` are no longer written. The generic
`for h in ALL_HEADS: if h in heads: save_pt(getattr(mod, h)...)` loop must skip
`"fusion"` (it is a `ModuleDict`, handled explicitly above).

### 3.5 `gan` stage required files

`_run_local` (stage `gan`) and `_run_remote` (stage `gan`) currently require
`kw.pt` + `fusion.pt` in `--pt`. Replace both with
`fusion_gate.pt` + `fusion_gain.pt` + `fusion_mix.pt`.

### 3.6 Modal dispatch plumbing

`train_remote` / `_run_remote` carry per-`.pt` `bytes` kwargs for the `gan` stage.
Replace `kw_bytes` / `fusion_bytes` with `fusion_gate_bytes` / `fusion_gain_bytes`
/ `fusion_mix_bytes` in: the `train_remote` signature, its `uploads` dict, the
`_run_remote` `pt_bytes` dict, and the `stage == "gan"` existence check + read
block. Mechanical rename against the new `files.py` constants.

---

## 4. `model/data.py` and `model/config.py`

### 4.1 `data.py`

- Delete the `FLEX_JSON` import, the `KW_VOCAB` / `FLEX_N` module-level block.
- `record` dataclass: `flex_tf` field -> `kw: list = field(default_factory=list)`.
- `read`: capture `d.get("kw") or []` onto the record (was `d.get("flex_tf")`).
- `_row_tf` -> `_row_kw(row) -> torch.Tensor` of shape `[len(EMOJIS)]`; scatter the
  sparse `[[emojiIdx, value], ...]` pairs into a dense float tensor.
- `EmojiDataset`: `self.flex_tf` -> `self.kw = torch.stack([_row_kw(r) for r in records])`;
  `__getitem__` returns `(text, emoji, style, colors, kw)` (arity unchanged).
- If a `fusion` run hits a dataset whose rows have no `kw` field (regen run with
  `--no-kw`), fail fast with a clear message from `_train_encoder` (check the first
  train record) rather than silently training on all-zero `kw`.

### 4.2 `config.py`

- Remove `DROPOUT_KW`. No new hyperparameters (the `FusionHeadGate` BN has none
  worth exposing; `INFONCE_TEMP` is reused).
- Anything importing `FLEX_N` from `model.data` switches to `len(EMOJIS)`.

---

## 5. Report (`tools/report.py`)

### 5.1 Model -> Emojis section

The `acc@k` (k 1..10) overlay on `data/eval.jsonl` becomes **5-way**:

1. `EmojiHead` — `emb.score(emoji_head(enc(text)))`
2. `Keywords` — the row's precomputed `kw` dense vector, ranked descending
3. `Fusion·Gate` — `FusionHeadGate` loaded from `pt/fusion_gate.pt`, on
   `(emoji_logits.detach(), kw)`
4. `Fusion·Gain` — `pt/fusion_gain.pt`
5. `Fusion·Mix` — `pt/fusion_mix.pt`

`kw` per eval row comes from `model.data._row_kw`. A missing `fusion_*.pt` or a
shape mismatch drops that curve (not fatal), consistent with the existing
`.pt`-optional behaviour. The CLDR baseline overlay stays.

The second chart (the `data/keywords.json` probe) and the CLDR probe chart stay
**EmojiHead-only + baseline** — those probe strings carry no `kw` payload and we do
not recompute Fuse in Python.

### 5.2 Model -> Keyword vocab section

Kept as a pure diagnostic (every qualifying `data/ii.json` key scored by
`EmojiHead` rank). Its `query_tokens` import moves from `model.flexrank` to
`model.tokenize`. `report.json.keywords_flex.ranked` **no longer feeds `regen`**
(the `kwvocab.ts` consumer is deleted) — note this in the section text.

### 5.3 `report.json`

`report.json.emoji` gains `keywords`, `fusion_gate`, `fusion_gain`, `fusion_mix`
`acc@k` arrays alongside the existing `emojihead` / baseline arrays.

---

## 6. ONNX export + web

### 6.1 `model/export_onnx.py`

- `ExportWrapper.forward(x)` — **one input**. Outputs
  `(style_logits, emoji_logits, color)` — **three**. Drop the `flex_tf` input,
  `kw` / `KWHead`, `fusion` / `FusionHead`, and the `kw_logits` / `fusion_logits`
  outputs. `export_onnx` `input_names` / `output_names` / `dynamic_axes` updated.
- `FUSION_EXPORT_VARIANT = "gate"` module constant. `export()` loads
  `pt/fusion_<variant>.pt` and serialises its parameters into `meta.json`:

  ```json
  "fusion": {
    "variant": "gate",
    "bn_mean": [...6...], "bn_var": [...6...],
    "w": [...6...], "b": <float>
  }
  ```

  For `gain`: `{ "variant": "gain", "beta": <softplus(raw_beta)> }`.
  For `mix`: `{ "variant": "mix", "g": <sigmoid(raw_g)> }`.
- `meta.json`: drop `flex_kw` / `flex_n`, add the `fusion` block.
- `export_web` does not write `keywords.json` (section 1.6 — `regen.ts` owns it).
  Drop the `FLEX_N` / `KW_VOCAB` imports from `model.data`.

### 6.2 `web/`

- `web/src/keywords.js` — `makeKeywordPredictor(kwprojJson, emojiCount)`: build the
  uFuzzy instance + haystack once; `predict(normText) -> Float32Array(emojiCount)`
  via `normText.split(" ")` (filter `length >= 3`) -> per-word `uf.filter` /
  `uf.info` -> aggregate (section 1.3-1.4). The caller passes the already-`normalize`d
  text (`useOnnx.js` has `char2idx` and calls `normalize` for `encode` anyway).
- `web/src/fusion.js` — `makeFusion(meta.fusion)`: `fuse(emojiLogits, kwArr) ->
  Float32Array(957)` implementing the exported variant's formula on **raw logits**
  (`z()` computed in JS: mean/std over the array). Only `gate` needs to be complete
  for the first ship; `gain` / `mix` are a few lines each and included.
- `web/src/hooks/useOnnx.js`:
  - fetch `kwproj.json` instead of `flex.json`; build `makeKeywordPredictor`
    instead of `makeFlexRanker`.
  - `predict`: no `flex_tf` tensor; run the session with `input` only. Compute
    `kwArr = keywordPredictor.predict(normalize(text, char2idx))`. Return
    `{ feeling, emoji: sigmoid(emoji_logits), kw: kwArr,
       fusion: fusionFn(emoji_logits.data, kwArr), palettes, ms }`.
    (`emoji_logits.data` raw for the fusion input; `emoji` stays sigmoid for the
    `model` toggle; `kw` raw for the `keywords` toggle. Ranking only needs order.)
- `web/src/App.jsx` — `EMOJI_MODES` / `pickEmojiList` / masthead toggle unchanged
  in shape (`fusion` default / `model` / `keywords`); all three now come from the
  `scores` object above.
- `web/package.json` — add `@leeoniya/uFuzzy`, remove `flexsearch`.
- Root `package.json` — add `@leeoniya/uFuzzy` (used by `regen.ts`), remove the
  `flexsearch` devDependency (only `tools/data/flexrank.ts` used it).

---

## 7. Deletions and file inventory

### Delete

- `model/flexrank.py`, `model/test_flexrank.py`
- `tools/data/flexrank.ts`, `tools/data/kwvocab.ts`
- `web/src/flexrank.js`, `web/src/flexrank.test.js`, `web/src/flexrank.fixture.json`
- `web/public/flex.json`
- `FLEX_JSON` (`files.py` + `files.ts`); `FLEX_FIXTURE_JSON` (`files.ts`)
- `KW_PT`, `FUSION_PT` (`files.py`) -> replaced by `FUSION_GATE_PT`,
  `FUSION_GAIN_PT`, `FUSION_MIX_PT` (`files.py`)
- add `KWPROJ_JSON = <web_public>/kwproj.json` to **both** `files.py` and
  `files.ts` (`regen.ts` writes it, `export_onnx.py` / `pred.py` / `report.py` do
  not read it — the browser does). `II_JSON` already exists in both.
- `regen.ts`: the `flexrank` / `kwvocab` imports, the report-vocab branch, the
  `FLEX_FIXTURE_TEXTS` fixture regeneration, `flex.json` write. Add the `proj`
  build (section 1.1) + `kwproj.json` write + per-row `kw` field. Keep `--no-kw`
  (now: skip the `kw` field and the `kwproj.json` write). `--matrix` /
  `--analysis` unaffected.

### New

- `model/tokenize.py`
- `tools/data/tokenize.ts`
- `web/src/tokenize.js`
- `web/src/keywords.js`, `web/src/keywords.test.js`
- `web/src/fusion.js`, `web/src/fusion.test.js`

### Modified

`model/model.py`, `model/train.py`, `model/data.py`, `model/config.py`,
`model/pred.py`, `model/export_onnx.py`, `model/test_model_heads.py`,
`tools/report.py`, `tools/data/regen.ts`, `files.py`, `files.ts`,
`web/src/hooks/useOnnx.js`, `web/src/App.jsx`, `web/package.json`,
`package.json`, `CLAUDE.md`.

### `model/pred.py`

- Drop the `FLEX_JSON` / `FlexRanker` / `KWHead` / `FusionHead` imports; import the
  exported variant's `FusionHead*` from `model.model` and read
  `FUSION_EXPORT_VARIANT` (or default `gate`).
- `fusion_top_labels` is emitted when `pt/fusion_gate.pt` exists (no `flex.json`
  dependency). `kw` for the 200 eval rows comes from `data/eval.jsonl` via
  `model.data._row_kw`; `fused = FusionHeadGate()(emoji_logits.detach(), kw)`.

### `model/test_model_heads.py`

Rewrite for the three heads:
- shapes: `FusionHeadGate/Gain/Mix()(randn(B, N), rand(B, N)).shape == (B, N)`
- init-neutral with `kw == 0`: `fused` equals the head's model-only baseline
  (`logit_m` for `gain`; `z(logit_m)` for `mix` at `g~=0.12` — assert close within
  `g`; `z(logit_m)` for `gate` at `a=0.5` — assert `0.5*z(logit_m)` ... i.e. assert
  the exact formula, not "matches EmojiHead")
- detach: `fused.sum().backward()` leaves `logit_m.grad is None` when `logit_m` is
  passed detached; the heads' own params get grad.
- `gate` `a` starts at `0.5`; `gain` `softplus(raw_beta)` starts at
  `softplus(0)~=0.693`; `mix` `g` starts at `~=0.12`.

---

## 8. Verification

Non-training:

- `uv run ruff check .` / `uv run ruff format --check .`
- `uv run python model/test_runmeta.py`
- `uv run python model/test_train_cli.py`
- `uv run python model/test_model_heads.py`
- `uv run python tools/test_report.py`
- `bun run regen` — confirm each train/eval row has a `kw` field; run twice and
  diff `data/train.jsonl` / `data/eval.jsonl` / `data/labels.json` to confirm
  determinism.
- `cd web && npm test` (adds `keywords.test.js`, `fusion.test.js`; removes
  `flexrank.test.js`) and `npm run build`.

Behavioural (full run, not a smoke test):

- `train enc --local` — watch `MRR/fusion_gate/val`, `MRR/fusion_gain/val`,
  `MRR/fusion_mix/val` against `MRR/e/val` in TensorBoard. Expected: each fusion
  variant `>= MRR/e/val` (fusion can only add precise keyword evidence on top of a
  detached model ranking); the best variant is the comparison winner.
- `uv run python model/pred.py --pt pt` — spot-check `data/pred.jsonl`
  `fusion_top_labels`.
- `uv run python tools/report.py` — the 5-way `acc@k` overlay renders; read off
  which `Fusion·*` curve dominates.

## 9. Open risk

Dropping `flexrank.fixture.json` removes the cross-language conformance guarantee.
Mitigation: the surviving cross-language surface is just the tokenizer (~6 trivial
lines, same class of hand-kept parity as `normalize` / `encode` already in the
tree) plus Fuse.js config constants; `web/src/keywords.test.js` pins a handful of
`text -> expected emoji` cases. The soft-TF fuzzy matcher that actually needed a
fixture is gone.
