# Keyword-based emoji search

_2026-09-12_

There is no runtime keyword search anymore — no inverted index, no fuzzy match, no
fusion combiner blending a lexical score with `EmojiHead`. `EmojiHead` is queried
directly on whatever text the user typed, in the browser and in training alike (see
`docs/model.md`). The web app ships no `keywords.js`, no `kwproj.json`, no
`fusion.js`, and the model has no `KWHead`/`FusionHead`/gate.

(An earlier version of this file described that system in full —
`git log docs/search.md` has it. It was removed project-wide; the vocab-coverage
gap it aimed to close is now covered by mixing CLDR/EmojiLib keyword text
directly into training, below.)

## What replaced it

`tools/data/keywords.ts` merges the CLDR (`data/cldr.jsonl`) and EmojiLib
(`data/emojilib.jsonl`) keyword→emoji sources, dedupes by normalized text, and
splits the result by word count into two committed files:

- **`data/keywords.jsonl`** — single-word rows (`wordCount(text) === 1`).
- **`data/terms.jsonl`** — everything else (multi-word phrases).

Only rows whose emoji survives the current trained vocab (`data/labels.json`) are
kept; rows shorter than 3 chars are dropped.

## Training-time mix-in (`model/data.py`)

`SAMPLING_SOURCES` (`model/config.py`) maps a source name to its `SamplingSource`
— the jsonl file to sample, the TB metric that judges it, and that metric's goal
(e.g. `"keyword": SamplingSource("data/keywords.jsonl", "acc@1", 0.90)`). Adding a
source, e.g. a future `flags.jsonl`, is one new entry here; nothing else in
`model/data.py` or `model/train.py` needs to change.

`EmojiDataset(mix_sources=True)` (used by `train_ds()` and the eval loader) loads
one pool per `SAMPLING_SOURCES` entry and, per training example, replaces it with
a row from one of those pools instead of the corpus row, at that source's
*current* rate:

```
r = rand()
walk SAMPLING_SOURCES in order, accumulating each source's current rate;
r under the running total → sample that source's pool
otherwise                 → the corpus row unchanged
```

All sources share one base/floor rate (`SAMPLING_BASE_RATE`, `SAMPLING_MIN_RATE`
in `model/config.py`) — there's no per-source static rate to tune, since the
per-source *goal* already differentiates them. Each source's current rate starts
at `SAMPLING_BASE_RATE` and is adjusted once per epoch in
`LitEncoder.on_train_epoch_start` (`model/train.py`), from that source's own
`cfg.metric` reading the previous epoch's validation, scaled by how far that
reading is from `cfg.goal`:

```
gap  = max(0, goal - metric@val) / goal        # 0 once the goal is met
rate = clamp(base_rate * gap, SAMPLING_MIN_RATE, base_rate)
```

so a source that has already met its goal falls back to `SAMPLING_MIN_RATE`
(never zero) and a source still short of goal keeps close to the full shared
base rate — `SamplingRates` (`model/data.py`) holds the live per-source rate,
mutated in place so the same `EmojiDataset` instance the `DataLoader` reads
picks up the change (only correct with `num_workers=0`; a shared-memory tensor
would be needed if training moved to worker processes for `train_ds()`'s loader).
These rows train `EmojiHead`/`StyleHead` exactly like any other row (same
`lse_infonce` loss) — there is no separate lexical loss or head.

## Measuring it (`tools/report.py`)

Two report sections probe `EmojiHead` directly on these files (bare retrieval, no
combiner):

- **`keyword`** — `EmojiHead` retrieval over `data/keywords.jsonl` (priority-1 goal,
  `emoji prediction.keyword.acc@{1,5,10}` in `goals.yml`).
- **`term`** — same probe over `data/terms.jsonl` (priority-2 goal,
  `emoji prediction.term.acc@{1,5,10}`).

`emoji.eval` (priority-3) is the same `EmojiHead` retrieval over `data/eval.jsonl`'s
full-text rows — the untouched, non-keyword short texts.

## `data/ii.json`

Still present, but only as an intermediate in `tools/data/emojilib.ts` (merging
new EmojiLib keyword→emoji pairs) and as the source for the `keywords_flex`
report diagnostic (`tools/report.py:_flex_keyword_candidates`) — a single-token
keyword-vocab coverage check, not a scoring path. Nothing reads it at inference
time.
