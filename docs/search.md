# Keyword search for emoji retrieval — library comparison

_2026-09-10_

## Problem

Retrieve a **scored, ranked list of emojis** from a short query string.

- **Documents:** ~3000 emojis, each a bag of single-word keywords
  (`{ emoji: "😀", keywords: ["happy", "smile", "joy"] }`). In this repo the
  source is `data/ii.json` — **5031** keyword keys → emoji lists — and the
  trained emoji vocab (`data/labels.json`) is **957** entries. Call it
  ~3000 docs / ~5000 unique keywords.
- **Query:** free text, `normalize`d, `≤ 42` chars, usually 1–6 words
  (`"I am happy today"`).
- **Hard requirements:** lightweight (small or zero bundle add), fast
  (sub-millisecond per query), runs in the browser (`web/` is Vite + React,
  ships static to Pages, `onnxruntime-web` wasm already loaded).
- **Repo-specific requirement:** the scorer must be **isomorphic**. The same
  scores are needed in three places today, kept byte-identical by a
  conformance fixture (`web/src/flexrank.fixture.json`):
  - `tools/data/regen.ts` (Bun) — precomputes a keyword signal for every
    train/eval row.
  - `web/src/flexrank.js` (browser) — live at inference time.
  - `model/flexrank.py` (Python) — for `tools/report.py` / `model/pred.py`.

  A pure-JS library covers the first two surfaces; the Python one must then be
  hand-ported or dropped.
- **Downstream:** the keyword scores are not shown raw — they feed the learned
  `KWHead` / `FusionHead` fusion with the neural `EmojiHead` (see
  `docs/model.md`). The output shape the fusion path wants is a dense
  `Float32Array(N)` over emoji-vocab order. A standalone "keywords only" UI mode
  wants a ranked `[emoji, score]` list.

### Why this is not quite a full-text search problem

Classic full-text engines (Lunr, FlexSearch, MiniSearch, Orama) assume
**documents are prose**: they tokenize and stem the document body, build an
inverted index over those tokens, and rank with TF-IDF/BM25 where term
frequency and document length carry signal.

Here every "document" is already a short list of **atomic, hand-curated
keywords**. Term frequency within a doc is almost always 1; document lengths
are nearly uniform (2–5 keywords). So the parts of BM25 that do the heavy
lifting on prose (TF saturation, length normalization) contribute almost
nothing. What actually matters:

1. **Query tokenization** — split, lowercase, drop stopwords.
2. **Term → keyword matching** — exact, plus prefix/stem/fuzzy for morphology
   (`running` → `run`, `cats` → `cat`).
3. **IDF-style weighting** — a keyword pointing at 1 emoji (`abacus`) is worth
   more than one pointing at 15 (`love`).
4. **Aggregation** — sum keyword weights per emoji.

Every option below is really a different way to do step 2. Steps 1, 3, 4 are
~30 lines of your own code regardless.

## Evaluation criteria

| Criterion | Weight | Notes |
|---|---|---|
| Bundle size added | high | stated requirement; `onnxruntime-web` already dominates, but every KB on the critical path counts |
| Query latency | high | must stay well under 1 ms; called on every keystroke |
| Index build time / memory | medium | 5k keywords is tiny; anything reasonable is fine |
| Isomorphic (Bun + browser, ideally Python) | high | three parity surfaces today |
| Retrieval quality (rough MRR) | medium | lexical ceiling is low; the neural head does the semantic lifting |
| Integration effort | high | how much new code + how many moving parts |

### On the MRR numbers below

These are **very rough** estimates for lexical-only retrieval against the
annotated emoji set on a held-out set of natural short texts. Caveats:

- "Relevant" = the row's annotated emojis (typically 0–3). Many natural
  queries share **no** keyword with any doc ("ugh, Monday again") — lexical
  methods score 0 on those, which caps MRR hard.
- The spread between competent lexical libraries is small (~0.05). The big
  levers are **stemming/lemmatization**, **synonym expansion**, and **fusion
  with the neural head** — not the library.
- Numbers assume the query-side tokenizer already drops stopwords (as
  `web/src/flexrank.js:queryTokens` does).

## Candidates

| Option | Bundle (min+gz) | Query latency¹ | Isomorphic | Rough MRR² | Integration |
|---|---|---|---|---|---|
| **Hand-rolled inverted index + IDF** | **0 KB** | ~5–20 µs | JS + Python (you own it) | 0.30–0.42 | ~60 LOC, already ~80% built |
| **MiniSearch** | ~6 KB | ~30–80 µs | JS only | 0.33–0.45 | ~40 LOC glue, drop Python parity |
| **uFuzzy** (matcher only) | ~5.5 KB | ~50–150 µs | JS only | 0.32–0.44 (as a matching layer) | ~50 LOC glue; you still write IDF + aggregation |
| **FlexSearch 0.8** | ~17 KB | ~20–60 µs | JS only | 0.33–0.45 | ~50 LOC; heavy API, already a dead dep |
| **Lunr.js** | ~8.7 KB | ~40–100 µs | JS only (Python: none) | 0.34–0.46 (stemmer helps) | ~40 LOC; prebuilt index, stemmer pipeline |
| **Fuse.js** | ~5.5 KB | ~1–4 ms | JS only | 0.22–0.34 | ~30 LOC; wrong tool, slow, no IDF |
| **Orama** | ~30–45 KB | ~40–100 µs | JS only | 0.33–0.45 | ~40 LOC; overkill at this scale |
| **wink-bm25-text-search** | ~4 KB + tokenizer ~5 KB | ~50–120 µs | JS only | 0.34–0.46 | ~50 LOC; needs wink tokenizer/stemmer |

¹ Per query, ~5k keywords, warm. Order-of-magnitude only.
² Lexical-only, annotated-emoji ground truth, natural short-text eval. See caveat above.

---

### 1. Hand-rolled inverted index + IDF weighting — **recommended**

**How it works.** Build once: `Map<keyword, emojiIdx[]>` straight from
`data/ii.json`, plus `weight[keyword] = 1 / log2(1 + df)` where `df` is how many
emojis the keyword points at (cheap IDF proxy). Per query: `queryTokens(text)`
(lowercase, strip punctuation, split, drop stopwords, len ≥ 2), then for each
token look up the keyword map — exact hit, plus optionally a prefix scan or a
length-ratio fuzzy match for morphology (this is exactly what
`web/src/flexrank.js:overlap` already does). Accumulate
`score[emoji] += match_strength * weight[keyword]`. Return the dense
`Float32Array(N)` or a sorted list.

**Performance.** Fastest option — a few `Map` lookups per token, no library
init, no scan of the full keyword list unless you opt into prefix/fuzzy. Index
is two `Map`s: ~5k string keys + ~10k small int arrays, well under 1 MB heap.
Build time is a single pass over `ii.json` (~1 ms).

**Rough MRR: 0.30–0.42.** Exact + prefix + the current length-ratio fuzzy.
Add a stemmer (e.g. a 30-line Porter or a hand list of `-ing/-ed/-s` strips)
to push toward the top of that range. This is the current
`flexrank.js` design; its known weakness is pure paraphrase with no shared
stem, which is precisely what the neural head is there to cover.

**Integration.** Lowest — you already have `queryTokens` / `overlap` /
`tfVec` in `web/src/flexrank.js`, a Bun copy in `tools/data/flexrank.ts`, and a
Python copy in `model/flexrank.py`, all locked by
`web/src/flexrank.fixture.json`. Swapping the soft-TF-vector output for a
direct `[emoji, score]` aggregation is a localized change to those three files
plus the fixture regen in `regen.ts`. Zero new dependencies, zero bundle
delta, and it is the **only** option that keeps the Python parity surface
without a hand-port.

**Downside.** You maintain the matching logic. But it is ~60 lines and already
written.

---

### 2. MiniSearch — best library option

**How it works.** In-memory inverted index with BM25 scoring, prefix search,
and configurable fuzzy (edit-distance) matching. You'd index one document per
emoji with a single `keywords` field (space-joined), give it
`processTerm` for lowercasing/stemming, then `search(query, { prefix: true,
fuzzy: 0.2, combineWith: 'OR' })` and map result IDs → emoji. Its `boostDocument`
hook or a post-pass gives you the IDF-ish reweighting; BM25's own IDF already
does most of that.

**Performance.** ~6 KB gzip. Index build for 3k tiny docs is a few ms; heap a
few hundred KB. Query latency tens of µs. Serializable index
(`MiniSearch.loadJSON`) if you want to skip the build, though the build is
cheap enough to skip that.

**Rough MRR: 0.33–0.45.** BM25 IDF + prefix + fuzzy is a solid lexical
baseline and slightly better out of the box than a naive hand-rolled exact
match, mostly from the built-in fuzzy and prefix. With a real stemmer in
`processTerm` it reaches the top of the range. BM25 length-norm buys ~nothing
here (uniform doc lengths).

**Integration.** ~40 lines of glue in `web/` and a parallel ~40 in
`regen.ts`. New dep in both `web/package.json` and root `package.json`.
**Breaks the Python parity surface** — `model/flexrank.py` has no equivalent,
so `tools/report.py` / `model/pred.py` either re-implement MiniSearch's BM25 +
tokenizer in Python (non-trivial, easy to drift) or drop the exact-parity
requirement and treat the report's keyword diagnostic as approximate. The
in-flight "simple fusion" plan already chooses to drop the Python ranker, so
this cost may be acceptable.

**When to pick it.** If you want a maintained library, readable config, and
don't want to own the matching code — and you're OK with JS-only parity.

---

### 3. uFuzzy — fuzzy matching layer, not a ranker

**How it works.** `@leeoniya/uFuzzy` matches a short needle against a flat list
of short haystack strings ("a more forgiving `String.includes()`"). Default
MultiInsert mode requires all needle chars in order with limited insertions;
it returns matched indices plus per-match char/gap info you turn into a
similarity. It has **no inverted index and no IDF** — you feed it your 5k
keyword list as the haystack, call it once per query word, and get back
matching keywords + a similarity you multiply by your own `weight[keyword]`
and aggregate per emoji. This is what the current "simple fusion"
implementation plan (`docs/superpowers/plans/2026-09-10-simple-fusion-three-fusionheads.md`)
switched to (from Fuse.js).

**Performance.** ~5.5 KB gzip. No build step. Per query word it does an
`O(haystack)` filter over 5k strings then an `info` pass over the survivors —
low hundreds of µs for a few-word query. Memory trivial (holds the haystack
array).

**Rough MRR: 0.32–0.44** used as the matching layer under your own IDF +
aggregation. Its strength over the current prefix/length-ratio `overlap` is
tolerance of internal edits and transpositions (`recieve`→`receive`,
`freind`→`friend`) and out-of-order intra-word matching; its risk is
over-matching short words (mitigated with `intraIns`/similarity floors, as the
plan does: `{ intraIns: 1 }`, `sim >= 0.5`, skip words `< 3` chars).

**Integration.** ~50 lines: you still write `queryTokens`, `weight`, and the
per-emoji accumulation — uFuzzy only replaces the inner `keyword ~ word`
test. New dep in `web/` and root. Python parity: none — same situation as
MiniSearch. Because you keep the tokenizer + IDF + aggregation, the browser
and Bun copies stay small and easy to hold identical.

**When to pick it.** If the main quality gap you see is typo/morphology
tolerance on the keyword match, and you want that without hand-writing a
stemmer. It is a strict upgrade to `flexrank.js:overlap` at ~5.5 KB.

---

### 4. FlexSearch 0.8 — already a (dead) dependency

**How it works.** High-performance inverted index with contextual/lexical
indexing, configurable tokenizers (`strict`/`forward`/`full`), optional
built-in encoders (case folding, simple stemming, phonetic). Index one doc per
emoji on a joined `keywords` field; `index.search(query, { suggest: true })`
returns ranked IDs. Scoring is a proprietary resolver, not textbook BM25;
there is no first-class per-term IDF hook, so keyword-rarity weighting is a
post-pass on your side.

**Performance.** ~17 KB gzip (the largest of the "small" set; a stripped custom
build is smaller but a build-config chore). Query latency is excellent (tens
of µs). Build + heap for 3k tiny docs is negligible.

**Rough MRR: 0.33–0.45.** Comparable to MiniSearch. The `forward`/`full`
tokenizers give cheap prefix/substring recall; the built-in stemmer is weak
but free.

**Integration.** ~50 lines. It's **already in `web/package.json` and root
`package.json`** (`flexsearch@0.8.212`) but currently **unused** — the
`flexrank.*` files are hand-rolled despite the name. So "adopting FlexSearch"
means actually wiring it in for the first time. API surface is large and the
docs are famously terse; expect trial and error on tokenizer/resolver options.
Python parity: none. Given it's 3× the size of MiniSearch for no quality gain
at this scale, prefer MiniSearch unless you specifically want FlexSearch's
tokenizer modes.

---

### 5. Lunr.js / elasticlunr

**How it works.** Build-time inverted index (`lunr(builder => …)`), BM25-ish
scoring, a token pipeline with an **English Porter stemmer** and stopword
filter built in. Query `idx.search("happy today")` (supports `term~1` fuzzy,
`term*` wildcard, field boosts). Serialize the index to JSON and ship it.

**Performance.** ~8.7 KB gzip (lunr) / ~7 KB (elasticlunr). Index build is
meant to be offline — do it in `regen.ts`, ship the JSON, `lunr.Index.load()`
in the browser (fast). Serialized index for 3k tiny docs is maybe 200–400 KB
JSON — bigger than shipping `ii.json` itself (272 KB), so a mild negative.
Query latency tens–hundred µs.

**Rough MRR: 0.34–0.46** — the built-in Porter stemmer is the reason it edges
the others; `running`/`ran`/`runs` all collapse to one term on both sides.

**Integration.** ~40 lines, but two runtimes: build in Bun, load in browser,
and the serialized-index format is a version-locked artifact you must
regenerate whenever `ii.json` changes (fits the existing `regen` +
`web/public/` pattern). Fuzzy is edit-distance only (no prefix-ratio). Python
parity: none, and Lunr's pipeline is annoying to reproduce exactly. Default
config is opinionated (drops short tokens, always stems) — occasionally
surprising for single-word keyword docs.

**When to pick it.** If you want stemming for free and are willing to ship a
prebuilt index artifact.

---

### 6. Fuse.js — not recommended

**How it works.** Bitap fuzzy string matching over a list. With
`keys: ['keywords']` it scores each emoji doc by best fuzzy substring match of
the **whole query string** (or per-token if you loop). No inverted index — it
scans every doc every query. Score is a normalized edit-distance blend; **no
IDF, no term rarity**.

**Performance.** ~5.5 KB gzip but **1–4 ms per query** at 3k docs (linear scan
with bitap per field) — 10–100× slower than everything else here, and it's on
the keystroke path. Memory fine.

**Rough MRR: 0.22–0.34.** Without IDF, a query word that fuzzy-matches a
common keyword (`love`, `time`) drags in dozens of unrelated emojis with no
down-weighting. Threshold tuning trades recall for precision but can't add the
rarity signal. This is why the in-flight plan **moved off Fuse.js to uFuzzy**
(commit `f3e3663`).

**When to pick it.** Don't, for this. It's built for "fuzzy-filter a list of
titles," not weighted retrieval.

---

### 7. Orama

**How it works.** Modern full-text (BM25) + optional vector/hybrid search,
typed schema, plugins. Index one doc per emoji; `search(db, { term, ... })`.

**Performance.** ~30–45 KB gzip (tree-shakeable, but the full-text path pulls a
fair amount). Fast queries, fast build. Overkill footprint for a 3k-doc bag-of-
keywords problem.

**Rough MRR: 0.33–0.45** — same lexical ceiling as MiniSearch/FlexSearch.
The vector/hybrid mode could in principle lift recall on paraphrase, but you
already have a purpose-built neural head for that; running a second embedding
model in the browser to feed Orama makes no sense here.

**When to pick it.** If you later want a single library doing lexical + vector
hybrid and can afford the bytes. Not now.

---

### 8. wink-bm25-text-search

**How it works.** Pure BM25 engine. You supply the tokenizer/normalizer
(typically `wink-nlp-utils`: tokenize, lowercase, remove stopwords, stem).
`addDoc` per emoji, `consolidate()`, then `search(query)` → ranked `[id,
score]`. Clean, explicit BM25 with proper IDF.

**Performance.** ~4 KB for the engine + ~5 KB for `wink-nlp-utils`. Build and
query both fast at this scale. Heap modest.

**Rough MRR: 0.34–0.46** — textbook BM25 IDF + a real stemmer via wink. On par
with Lunr, more transparent, no prebuilt-index artifact.

**Integration.** ~50 lines, two deps (engine + utils) in `web/` and root. You
control the pipeline, so browser/Bun parity is easy to hold. Python parity:
none, but BM25 + a Porter stemmer is at least a well-specified thing to
re-port if you must. More assembly than MiniSearch for roughly the same
result.

---

## Recommendation

**Keep it hand-rolled: inverted index + IDF weighting, with uFuzzy as an
optional drop-in for the fuzzy match step.**

Reasoning:

1. **The corpus doesn't reward a full-text engine.** Docs are atomic keywords,
   uniform length, TF ≡ 1. BM25's machinery is inert here; you'd be shipping
   6–45 KB to get `Map.get` plus a fuzzy matcher you could inline.
2. **You already have ~80% of it** (`flexrank.js` / `flexrank.ts` /
   `flexrank.py` + fixture). The remaining work is changing the output from a
   soft-TF vector to a direct per-emoji score aggregation.
3. **It is the only option that preserves the Python parity surface** without a
   hand-port. Every library is JS-only; adopting one forces
   `model/flexrank.py` to be re-implemented or the report's keyword diagnostic
   to go approximate.
4. **Lightest and fastest**, which are the two stated hard requirements: 0 KB
   added, single-digit-µs queries.
5. **Quality difference is marginal** (~0.05 MRR) and dominated by
   stemming/synonyms, which you can add to the hand-rolled tokenizer directly —
   and by the neural fusion, which already exists.

**If** you want a maintained library and are willing to drop exact Python
parity: **MiniSearch** is the pick — smallest real full-text lib, sane API,
BM25 + prefix + fuzzy in ~40 lines.

**Avoid** Fuse.js (slow, no IDF), FlexSearch (3× the size for no gain here,
awkward API), and Orama (footprint) for this specific problem.

### Concrete next step for the hand-rolled path

Extend the existing ranker instead of the soft-TF vector:

```
buildIndex(iiJson):
  postings: Map<keyword, emojiIdx[]>        # from ii.json, filtered to EMOJIS vocab
  weight:   Map<keyword, 1/log2(1+df)>      # df = postings[keyword].length

score(text) -> Float32Array(N):
  out = zeros(N)
  for word in queryTokens(normalize(text)):        # existing tokenizer
    for (keyword, strength) in matches(word):       # exact=1.0; prefix/stem/uFuzzy < 1.0
      w = strength * weight[keyword]
      for e in postings[keyword]: out[e] = max(out[e], w)
  return out
```

- `matches(word)`: start with exact + the current `overlap` prefix-ratio rule.
  If typo tolerance matters, replace the inner test with
  `uFuzzy({ intraIns: 1 })` + `sim >= 0.5` (skip words `< 3` chars) — a ~5.5 KB
  add that only touches `matches`, leaving tokenizer/IDF/aggregation identical
  across the JS and Bun copies.
- Keep the three-surface parity discipline: update `flexrank.js` /
  `flexrank.ts` / `flexrank.py` together and regenerate
  `web/src/flexrank.fixture.json` in `regen.ts`. If you adopt uFuzzy, the
  Python copy either ports a minimal MultiInsert matcher or the report's
  keyword diagnostic is documented as approximate (the in-flight
  "simple fusion" plan already takes the latter route).

## Sources

- [flexsearch vs fuzzysearch vs minisearch — npm trends](https://npmtrends.com/flexsearch-vs-fuzzysearch-vs-minisearch)
- [flexsearch vs minisearch (2026) — devpick](https://devpick.co/flexsearch-vs-minisearch)
- [fuse.js vs minisearch (2026) — devpick](https://devpick.co/fuse.js-vs-minisearch)
- [elasticlunr vs flexsearch vs fuse.js vs minisearch — npm-compare](https://npm-compare.com/elasticlunr,flexsearch,fuse.js,minisearch)
- [@leeoniya/uFuzzy — GitHub](https://github.com/leeoniya/uFuzzy)
- [@leeoniya/ufuzzy — npm](https://www.npmjs.com/package/@leeoniya/ufuzzy)
- [(Fuzzy) Search on The Frontend — beathagenlocher.com](https://beathagenlocher.com/fuzzy-search-on-the-frontend/)
