# Keyword-based emoji search

_2026-09-10_

This describes the keyword predictor as it is actually implemented today: a
non-learned, hand-rolled inverted index with IDF weighting and a fuzzy match
step. It takes a short text and produces a score for every emoji in the trained
vocab. (An earlier version of this file compared full-text libraries;
`git log docs/search.md` has it. The conclusion was to keep it hand-rolled, and
that is what shipped.)

## What it is for

Given a normalized query string (`≤ 42` chars, usually 1–6 words) return a dense
`Float32Array(N)` — one score per emoji, in `data/labels.json` order — that says
"how strongly does the wording of this text point at each emoji". The scores are
purely lexical. They are never shown raw as the product's answer; they feed the
learned fusion combiners next to the neural `EmojiHead` (see the end of this
doc), and a "keywords" debug mode in the masthead ranks emojis by this vector
directly.

## Inputs

**`data/ii.json`** — the committed inverted index. `{ "<keyword>": ["😀", "😄",
…], … }`, ~5031 single-word keyword keys, each mapping to the list of emoji
glyphs that keyword describes (CLDR-style annotation keywords → emoji). This is a
static artifact; nothing in the tree rebuilds it.

**`data/labels.json` `emojis`** — the dynamic trained emoji vocab (whatever
cleared `regen`'s frequency cap). Only emojis in this set can score; the index is
projected onto it.

## Build step (`tools/data/regen.ts`)

Run offline by `bun run regen`. Two derived structures:

1. **`proj`** — `ii.json` with every emoji list filtered to the current vocab and
   each glyph replaced by its vocab index. Keys whose emoji list becomes empty
   after filtering are dropped. So `proj: { "<keyword>": [<emojiIdx>, …] }`.

2. **`weight`** — an IDF proxy per surviving keyword:

   ```
   weight[k] = 1 / log2(1 + df)      df = proj[k].length
   ```

   A keyword pointing at one emoji (`df = 1` → `weight = 1.0`) counts for much
   more than one pointing at twenty (`df = 20` → `weight ≈ 0.23`). There is no
   term frequency and no document-length term — every "document" is an atomic
   keyword, so TF is always 1 and BM25's machinery would be inert.

`regen` writes `proj` to **`web/public/kwproj.json`** (`{ "proj": { … } }`) for
the browser. It does **not** ship `weight` — the browser recomputes it from
`proj` with the identical formula.

## Query time

### 1. Tokenize (`queryTokens`)

Ported verbatim to three files — `web/src/tokenize.js`, `tools/data/tokenize.ts`,
`model/kwtokens.py` — and must stay byte-identical:

```
text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')   // punctuation → space
    .split(/\s+/)
    .filter(w => w.length >= 2 && !STOPWORDS.has(w))
```

`STOPWORDS` is a fixed ~60-word list (`a an the to of in on at is it its i you we
they he she this that for and or but not with my your me am are was were be been
being do does did have has had will would can could just so if`). Tokens shorter
than 2 chars and stopwords are dropped; everything else is a query token.

### 2. Match each token to keywords (`matches(word)`)

For each query token, `matches` yields `(keyword, strength)` pairs:

- **Exact hit.** If `proj[word]` exists, yield `(word, 1.0)`.
- **Fuzzy hits.** Skip if the token is `< 3` chars. Otherwise run
  `@leeoniya/uFuzzy` (`new uFuzzy({ intraIns: 1 })`) with the token as the needle
  and the full list of `proj` keys as the haystack:
  - `uf.filter(keys, word)` narrows to candidate keys (all needle chars present
    in order, at most 1 inserted char between consecutive needle chars).
  - `uf.info(...)` returns, per candidate, `chars` = how many needle chars
    actually matched.
  - similarity `sim = info.chars[i] / keyword.length`. Keep the candidate only if
    `sim >= 0.5`. The exact key (`k === word`) is skipped here since it was
    already yielded at strength `1.0`.
  - yield `(keyword, sim)` — so a fuzzy strength is always in `[0.5, 1.0)`.

`intraIns: 1` plus the `sim >= 0.5` floor plus the `< 3` char skip are what keep
short tokens from fuzzy-matching half the index. The fuzzy step buys tolerance of
typos, transpositions, and light morphology (`recieve`→`receive`,
`kitten`→`kitty`); it does **not** do stemming or synonyms.

### 3. Score and aggregate

```
out = zeros(N)
for word in queryTokens(text):
  for (keyword, strength) in matches(word):
    v = strength * weight[keyword]
    for e in proj[keyword]:
      out[e] = max(out[e], v)          // max, not sum
```

Key points:

- A single `(keyword, emoji)` contribution is `strength · weight[keyword]` — the
  match quality times the keyword's rarity. Max value is `1.0` (exact hit on a
  `df = 1` keyword).
- Aggregation is **`max`**, both across the keywords a token matched and across
  the query tokens. An emoji's final score is the single strongest
  `strength · weight` any part of the query produced for it — matches do not add
  up, and a query with more tokens does not inflate scores.
- Emojis no token points at stay `0`. Many natural queries (`"ugh, monday
  again"`) share no keyword with any emoji and score all-zero — the lexical
  method has nothing to say, and fusion leans entirely on the neural head there.

### Output shape

- **Browser** (`web/src/keywords.js:makeKeywordPredictor(...).predict`): the
  dense `Float32Array(N)`.
- **`regen`**: a sparse per-row field `kw = [[emojiIdx, value], …]` — `value`
  rounded to 3 dp, `> 0` only, sorted by `emojiIdx` — written onto every
  `data/train.jsonl` / `data/eval.jsonl` row.

## The three surfaces and parity

| Surface | Role | How it gets the score |
|---|---|---|
| `tools/data/regen.ts` | offline, one score vector per train/eval row | builds `proj` + `weight`, runs the matcher, writes the sparse `kw` field + `kwproj.json` |
| `web/src/keywords.js` | live, per keystroke | loads `kwproj.json`, recomputes `weight`, runs the identical matcher |
| `model/data.py` | training / `tools/report.py` / `model/pred.py` | **does not re-run the matcher** — `_row_kw` just scatters the precomputed `kw` field into a dense `[len(EMOJIS)]` tensor |

So the actual matching logic exists only in JS/TS (the `regen` and browser
copies, which must stay char-identical — same `queryTokens`, same
`uFuzzy({ intraIns: 1 })`, same `sim >= 0.5`). The Python side is a passive
consumer of `regen`'s output; `model/kwtokens.py` carries `query_tokens` only for
an off-path keyword-vocab diagnostic in `tools/report.py`.

## Where the score goes

The dense `kw` vector is one of two inputs (the other being the model's raw
`emoji_logits`) to three tiny **detached** fusion combiners in `model/model.py`
— `FusionHeadGate`, `FusionHeadGain`, `FusionHeadMix`. They learn how much to
trust the keyword signal per text and blend it with the neural score;
"detached" means fusion training never perturbs the encoder or `EmojiHead`. The
winning combiner's parameters are exported into `meta.json` and re-run in the
browser by `web/src/fusion.js`. The masthead emoji-ranking toggle exposes all
three signals: `fusion` (default), `model` (raw logits), `keywords` (this `kw`
vector ranked directly).

## Properties and limits

- **Fast and tiny.** A few `Map` lookups per token plus one uFuzzy pass over ~5k
  short strings per token; sub-millisecond, no bundle cost beyond uFuzzy
  (~5.5 KB).
- **Deterministic.** Same text + same `ii.json` + same vocab → same vector,
  in `regen` and the browser alike.
- **No semantics.** Exact + fuzzy string match only. No stemming, no
  lemmatization, no synonyms, no paraphrase. `"running"` will not reach a `"run"`
  keyword unless uFuzzy's subsequence rule happens to bridge them. Paraphrase
  recall is explicitly left to the neural head via fusion.
- **`max` aggregation** means the score is a confidence in the best single
  lexical cue, not an accumulation of evidence — intentional, so that a wordy
  query cannot outweigh a terse one.
