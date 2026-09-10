# Report-driven keyword vocab + end-to-end FusionHead

Date: 2026-09-10
Status: approved for planning

## Summary

Two coupled changes to the emoji/keyword fusion pipeline:

1. **Keyword vocab selection** moves from the static CLDR `df == 1` filter in
   `tools/data/flexrank.ts` to a model-driven selection: the 1000 `data/ii.json`
   keywords that a trained `EmojiHead` ranks *worst* (target emoji furthest down
   its prediction list). `tools/report.py` computes the ranking and emits it into
   `report.json`; `tools/data/regen.ts` reads the newest report and builds
   `flex.json` from its top 1000. A no-report bootstrap path selects from
   `data/ii.json` deterministically.

2. **FusionHead training** becomes end-to-end. `loss/emoji` and `loss/kw` are
   removed as standalone losses when `fusion` is selected; only `loss/fusion`
   remains, and it backpropagates through the gate into `q_txt` (→ `EmojiHead` →
   encoder), `q_kw` (→ `KWHead`), and `EmojiEmbedding` — none of them detached.
   `FusionHead` bias init changes `4.0 → 0.0`. The mean gate value is logged.

Goal: let the two heads specialise. The vocab is exactly the region where the
text encoder is weak, and the single fusion loss with a per-row gate means
neither head is forced to solve the whole emoji-retrieval space alone.

## Motivation

Today `fusion` is a detached gate: `loss/fusion` trains only the
`Linear(TEXT_EMBED_SIZE + FLEX_N → 1)` gate, while `loss/emoji` and `loss/kw`
each independently push their head to solve the full task against
`EmojiEmbedding`. The `KWHead` input is the ~1388-key CLDR `df == 1` vocab —
chosen for IDF, not for where it helps. Result: `KWHead` and `EmojiHead` are
largely redundant and the gate has little to arbitrate.

New design: the `KWHead` vocab is the ~1000 keywords `EmojiHead` handles worst,
and the only training signal is the fused prediction. On a row `EmojiHead`
already nails, the loss is near-zero at `a → 1`, so `(1 - a) → 0` and `KWHead`
gets almost no gradient there. On a keyword-heavy row where `q_txt` is weak, the
loss can only drop by improving `(1 - a) · q_kw`, so `KWHead` gets the gradient
and the gate learns to route to it.

## Architecture: the feedback loop

```
regen ──► web/public/flex.json (kw vocab) ──► train enc ──► pt/enc.pt, pt/emoji.pt, pt/emoji_embed.pt
  ▲                                                                    │
  └──────────── report/<ts>/report.json (keywords_flex.ranked) ◄── tools/report.py
```

`regen`'s `flex_tf` vocab is no longer a pure function of `data/data.jsonl` +
flags; it depends on the newest `report/*/report.json`. `--matrix`,
`--analysis`, and `--no-kw` stay pure (they never read a report). The
`flexrank.fixture.json` conformance golden still locks whatever vocab was chosen,
and the three ranker surfaces
(`tools/data/flexrank.ts` / `model/flexrank.py` / `web/src/flexrank.js`) stay
byte-identical.

### Bootstrap (no usable report)

`regen` uses the report path only when `REPORT_DIR` contains a `report.json`
with a non-empty `keywords_flex.ranked`. Otherwise it uses the bootstrap path:

- Load `data/ii.json`.
- Keep key `k` when: `3 <= len(k) <= 6`; `k` matches `^[a-z0-9]+$`; `k` is not a
  `flexrank` stopword; and `ii.json[k]` shares at least one emoji with the
  current `data/labels.json` `emojis` vocab.
- Sort the surviving keys alphabetically, take the first 1000.

The CLDR `df == 1` build in `flexrank.ts:buildFlexRanker` is removed entirely —
it is not kept as a fallback.

### Steady state

`regen` reads the newest `report/*/report.json` (folder names sort lexically;
mirror `tools/data/upsample.ts:latestReportMedianLen`), takes
`keywords_flex.ranked[:1000]`, extracts the `kw` field, sorts alphabetically,
and feeds that list to the unchanged `overlap` / `tfVec` machinery. `FLEX_N =
min(1000, len(candidates))`.

Lifecycle on a fresh checkout:
`regen` (bootstrap) → `train enc` → `report.py` → `regen` (report) →
`train enc` → `report.py` → … converging once the worst-1000 set stabilises
between reports (expected 2–3 cycles).

## Candidate set and the rank metric

**Candidate keywords** (shared by the bootstrap filter and `report.py`): keys of
`data/ii.json` where `3 <= len(k) <= 6`, `k` matches `^[a-z0-9]+$`, `k` is not a
`flexrank` stopword, and `set(ii.json[k]) & set(EMOJIS)` is non-empty. The
single-token / alnum / non-stopword constraint is mandatory: `tf_vec`'s
`query_tokens` cannot emit a token for `"<3"`, `"1st place medal"`, a stopword,
etc., so those would be permanently-zero columns.

**Rank** for candidate `k` with in-vocab targets `T = set(ii.json[k]) & set(EMOJIS)`:

```
ids   = text_to_tensor(norm_text(k)).unsqueeze(0)
q_txt = EmojiHead(enc(ids))
order = EmojiEmbedding.score(q_txt).squeeze(0).argsort(descending=True)
rank  = min(1-based position of t in order for t in T)
```

Lower rank is better. Uses raw `EmojiHead` (`q_txt`), not the fused prediction.
Keywords whose `norm_text` is empty are skipped.

Candidates are sorted by `rank` **descending** (worst first); the first 1000
become the vocab.

## `tools/report.py` changes

- New `_flex_keyword_candidates()`: returns `{kw: [emojis ∩ EMOJIS]}` from
  `II_JSON` using the candidate filter above. `@cache`.
- New `_section_keywords_flex(enc, emoji_head)`: returns `{}` when `enc`,
  `emoji_head`, or `_emoji_embed()` is `None`. Otherwise batches the candidate
  texts through `enc` → `emoji_head` → `_emoji_embed().score`, computes each
  `rank`, and returns:

  ```json
  {
    "candidates": <int>,
    "missed": <count with rank > 10>,
    "ranked": [ {"kw": str, "emojis": [str], "top5": [str], "rank": int}, ... ]
  }
  ```

  `ranked` is the full candidate list sorted worst-first (regen needs all of it).
  `top5` is the top 5 emoji strings from `order`.
- Wire-up:
  - `build_report`: add `"keywords_flex"` to the default `want` set and to the
    `only` split; `report["keywords_flex"] = _section_keywords_flex(enc, emoji_head)`
    guarded by `"keywords_flex" in want`. It reuses the `enc` / `emoji_head`
    already loaded for `"emoji"`; ensure `need_enc` includes this section.
  - `_render_html`: new `_keywords_flex_html(d)` appended after `_emoji_html`.
    Renders `<h2>Model — Keyword vocab</h2>`, a summary line
    (`{candidates} candidates · {missed} missed (rank &gt; 10) · vocab = worst 1000`),
    and a table of **only** the `rank > 10` rows sorted by rank descending, with
    columns **Keyword · CLDR emojis · Top 5 predictions · Rank**. When `d` is
    `{}`, render a `.note` that `enc.pt` / `emoji.pt` / `emoji_embed.pt` are
    unavailable.
- `report.json` mirrors the section under `keywords_flex` (already automatic via
  the `report` dict).
- No change to the existing `keywords.json` probe or the CLDR probe — this is an
  added section.

## `tools/data/regen.ts` / `tools/data/flexrank.ts` changes

- `flexrank.ts`: delete the CLDR annotation load, the `df` map, and the
  `df == 1` filter. `buildFlexRanker` becomes `buildFlexRanker(kwVocab: string[])`
  — it takes an already-selected, already-sorted vocab and returns
  `{ kwVocab, tfVec, buildJson }` with the existing `overlap` / `tfVec` / `r3`
  logic unchanged. `queryTokens` import stays.
- `regen.ts`:
  - New `keywordVocabFromReport(): string[] | null` — scan `REPORT_DIR` newest
    first (`readdirSync` + `.sort().reverse()` like `upsample.ts`), read
    `report.json`, return `keywords_flex.ranked.slice(0, 1000).map(r => r.kw)`
    sorted alphabetically, or `null` if no report / no non-empty `ranked`.
  - New `keywordVocabFromIiJson(): string[]` — the bootstrap filter (reads
    `II_JSON` and `data/labels.json` emojis, applies the candidate filter,
    alphabetical, `slice(0, 1000)`).
  - In the `useKw` block: `const kwVocab = keywordVocabFromReport() ?? keywordVocabFromIiJson()`,
    then `const ranker = await buildFlexRanker(kwVocab)`. Everything downstream
    (`flex_tf` pairs, `flex.json` write, fixture write, `kwLine`) is unchanged
    except `kwLine` also reports the source
    (`source=report <folder>` / `source=ii.json bootstrap`).
  - `flexrank.ts` no longer needs `loadCldrAnnotations`; drop that import if now
    unused.
- `regen`'s stdout summary gains the vocab source on the `flex_tf` line.

## Training changes

### `model/model.py`

- `FusionHead.__init__`: `nn.init.constant_(self.net.bias, 4.0)` → `0.0`. Weight
  stays zero-init.

### `model/train.py` `_step` (the `if "fusion" in self.heads:` block)

Current:

```python
q_kw = self.kw(flex_tf)
kw_logits = self.emoji_embed.score(q_kw)
loss_kw = lse_infonce(kw_logits, emoji, INFONCE_TEMP)
loss = loss + loss_kw
self._log(f"loss/kw/{split}", loss_kw, bs)

a = self.fusion(enc, flex_tf).unsqueeze(-1)
q_fused = a * q_txt.detach() + (1 - a) * q_kw.detach()
w = self.emoji_embed.embed.weight.detach()
b = self.emoji_embed.bias.detach()
fusion_logits = q_fused @ w.t() + b
loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
loss = loss + loss_fusion
self._log(f"loss/fusion/{split}", loss_fusion, bs)
```

New:

```python
q_kw = self.kw(flex_tf)
kw_logits = self.emoji_embed.score(q_kw)          # diagnostic only

a = self.fusion(enc, flex_tf).unsqueeze(-1)
q_fused = a * q_txt + (1 - a) * q_kw              # no detach
fusion_logits = self.emoji_embed.score(q_fused)   # no detached w / b
loss_fusion = lse_infonce(fusion_logits, emoji, INFONCE_TEMP)
loss = loss + loss_fusion
self._log(f"loss/fusion/{split}", loss_fusion, bs)
self._log(f"gate/a/{split}", a.mean(), bs)
```

- `loss_kw` and `loss/kw/{split}` scalar are removed. `kw_logits` is still
  computed for `MRR/kw/{split}`.
- The `"emoji"` block is unchanged in structure, but when `"fusion" in
  self.heads`, `loss_emoji` must **not** be added to `loss`. Cleanest: in the
  `"emoji"` block, `loss = loss + loss_emoji` only when `"fusion" not in
  self.heads`; `emoji_logits` / `MRR/e/{split}` / `loss/e/{split}` logging stay.
  `q_txt = self.emoji(enc)` still runs (fusion consumes it, no detach).
- `MRR/e/val` and `MRR/kw/val` remain logged as diagnostics. Checkpoint /
  early-stop metric is unchanged: `MRR/fusion/val` when `fusion` is selected.
- `--heads` semantics: `fusion` still requires `emoji`. `train enc --heads emoji`
  (no `fusion`) is unchanged — `loss_emoji` is still added.
- `configure_optimizers` is unchanged: `emoji_embed.parameters()` is already
  included when `"emoji" in self.heads`, and it is now trained through the
  non-detached fused path.
- `.pt` outputs unchanged: `enc.pt`, `emoji.pt`, `emoji_embed.pt`, `style.pt`,
  `critic.pt` and — with `fusion` — `kw.pt`, `fusion.pt`.

### `model/config.py`

No new knobs. (`GATE_BALANCE_W` considered and rejected — logging `gate/a` is the
agreed observability.)

## What does not change

- `model/export_onnx.py`: still 2-input (`input`, `flex_tf`) / 5-output
  (`style_logits`, `emoji_logits`, `kw_logits`, `fusion_logits`, `color`). Only
  output *values* change. `FLEX_N` was already dynamic and already aborts export
  on a `kw.pt` shape mismatch.
- `model/pred.py`: unchanged (reads `flex.json` + heads, blends by the gate).
- `web/`: no code change — `FLEX_N` flows through `meta.json` `flex_n`;
  `flexrank.js` reads the regenerated `flex.json`; the 3-way masthead toggle is
  unchanged.
- `web/src/flexrank.fixture.json`: regenerated by `regen` in the same pass, as
  today.
- GAN stage (`LitColorGAN`, `energy/gan/*`): untouched.
- `bun run regen --analysis` / `docs/regen.md`: untouched (CLDR-overlap analysis
  does not use `flex_tf`).

## Files and constants

- New `II_JSON = "data/ii.json"` in both `files.ts` and `files.py`. `report.py`
  imports it from `files`; `regen.ts` / bootstrap import it from `../../files.ts`.
- `data/ii.json` is already git-tracked. No generator change in scope.

## Tests and verification

- `tools/test_report.py`: extend to assert the `keywords_flex` shape when heads
  are present, and that the section degrades to `{}` (no crash) when `.pt` files
  are absent.
- `model/test_train_cli.py`: assert `--heads style,emoji,critic,fusion` still
  parses and `--heads emoji` alone is accepted; assert `--heads fusion` without
  `emoji` still aborts.
- `uv run ruff check .` / `uv run ruff format --check .` clean.
- `uv run python model/test_runmeta.py`, `model/test_train_cli.py`,
  `tools/test_report.py` pass.
- `cd web && npm test` passes (regenerated `flex.json` + fixture).
- Behavioural (not a smoke test):
  1. `bun run regen` — confirm `flex_tf` line says `source=ii.json bootstrap`,
     `N <= 1000`.
  2. `train enc --local` — in TensorBoard watch `MRR/fusion/val` (checkpoint
     metric), `MRR/kw/val` climbing off zero (proves the gate is not collapsed),
     `MRR/e/val` (may dip — accepted), and `gate/a/train` settling below ~0.95.
  3. `uv run python tools/report.py` — new "Model — Keyword vocab" section
     renders; `report.json` has `keywords_flex.ranked`.
  4. `bun run regen` again — `flex_tf` line now says `source=report <folder>`.
  5. `uv run python model/pred.py --pt pt` — `fusion_top_labels` column present.

## Docs to update

- `CLAUDE.md`: the `regen.ts`, `flexrank.ts`, `report.py`, and `train.py`
  fusion/`_step` paragraphs. State explicitly that `regen`'s `flex_tf` vocab is
  no longer pure — it depends on the newest `report/*/report.json`, with an
  `ii.json` bootstrap when none exists; `--matrix` / `--analysis` / `--no-kw`
  stay pure. State that `loss/emoji` + `loss/kw` are dropped when `fusion` is on
  and only `loss/fusion` trains (end-to-end, no detach), with `MRR/e/val` /
  `MRR/kw/val` demoted to diagnostics and `gate/a/*` logged.
- `model.md` / `data.md`: background docs, update opportunistically (not gating).

## Risks and mitigations

1. **Gate collapse** — `a` saturates near 1 everywhere, `KWHead` never leaves
   zero-init, fusion == EmojiHead. Mitigated by the `4.0 → 0.0` bias init.
   Observability: `gate/a/{split}` and `MRR/kw/val`. If it still collapses, the
   fallback (not in this spec) is a `GATE_BALANCE_W * (a.mean() - 0.5).abs()`
   term or a brief `loss_kw` warmup.
2. **Vocab churn between runs** — each report re-targets the current encoder's
   weak spots, so `FLEX_N` and `kw.pt` / `fusion.pt` shapes can shift run-to-run.
   Accepted for v1 (`regen` already forces retrains on data/flag changes).
   Future: hysteresis (keep a keyword still in the worst ~1500) or freeze once
   stable.
3. **Weaker encoder emoji signal** — `MRR/e/val` is now a side effect. If the raw
   `model` toggle in the web app regresses badly, reconsider a small `loss_emoji`
   regulariser (explicitly out of scope here).
4. **Bootstrap vocab is not difficulty-filtered** — the first `KWHead` trains on
   an alphabetical slice; the strategy only engages from the second cycle.
5. **`regen` non-determinism** — documented in `CLAUDE.md`; the fixture golden
   still pins the chosen vocab, and the pure sub-commands are unaffected.

## Out of scope

- Vocab hysteresis / freezing.
- Any `GATE_BALANCE_W` knob.
- Ranking against fused logits instead of raw `EmojiHead`.
- A standalone `tools/rank_keywords.py` (kept in `report.py`).
- Changes to `data/ii.json`'s generator.
- `web/` code changes.
