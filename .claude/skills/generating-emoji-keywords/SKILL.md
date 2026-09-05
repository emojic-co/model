---
name: generating-emoji-keywords
description: Use when data/labels.json's emoji vocab has changed (after bun run regen) or data/keywords.json is missing/stale, and a keyword-to-emoji lookup index needs (re)building for keyword-based emoji tooling.
---

# Generating Emoji Keywords

## Overview

Builds `data/keywords.json`: a `keyword -> [emojis]` lookup index covering every emoji in `data/labels.json`. Keywords are generated directly by you, using your own knowledge of emoji semantics — this is a judgment task, not a deterministic transform, which is why it's a skill and not a `tools/data/*.ts` script that calls an external LLM API. `tools/report.py`'s keyword probe (in the Emojis section) reads this file, so it needs to stay committed and in sync with the current vocab.

## Steps

1. Read `data/labels.json`, take its `emojis` list (dynamic-size, changes on every `bun run regen`).
2. For each emoji, generate a short list of keywords strongly and specifically associated with it: lowercase English words, 1-4 per emoji, concrete objects/actions/concepts over vague ones (e.g. 🚀 → `["rocket", "launch"]`, 🥣 → `["bowl", "soup"]`, 🐕 → `["dog"]`, 🐶 → `["dog"]`). Every emoji must get at least one keyword — don't skip any. Keep this as an in-memory `emoji -> [keywords]` map.
3. Transpose it into `keyword -> [emojis]`: for each `(emoji, keywords)` pair, append the emoji to every keyword's list. A keyword legitimately maps to multiple emojis (e.g. `"dog" -> ["🐕", "🐶"]`). Dedupe each emoji list and sort it by the emoji's position in `data/labels.json`'s `emojis` list (stable, matches vocab frequency order).
4. Write the transposed map to `data/keywords.json` as a single flat JSON object — `{"<keyword>": ["<emoji>", ...], ...}` — pretty-printed, 2-space indent, keys sorted alphabetically, UTF-8, trailing newline. Not a JSON array, and not a list of `{"word": ..., "emojis": [...]}` records — see Quick reference below for the exact shape.
5. Report a one-line summary: emoji count in, keyword count out, 2-3 example entries.

## Notes

- `data/keywords.json` is derived from `data/labels.json`, but unlike `data/train.jsonl`/`data/eval.jsonl`/`data/labels.json` it is **committed to git** (like `data/data.jsonl`) — it doesn't need regenerating on every `bun run regen`, only when the vocab has actually drifted (new/dropped emoji) or the file is missing/stale. Never hand-edited; re-run this skill and commit the result instead.
- Do not shell out to `openai/gpt-5.6-luna` or any other API for this — you (the invoking model) are the keyword generator.

## Quick reference

Correct — one object, keyword keys:

```json
{
  "bowl": ["🥣"],
  "dog": ["🐕", "🐶"],
  "rocket": ["🚀"],
  "soup": ["🥣"]
}
```

Wrong — do not emit an array of `{word, emojis}` records:

```json
[
  {"word": "bowl", "emojis": ["🥣"]},
  {"word": "dog", "emojis": ["🐕", "🐶"]}
]
```
