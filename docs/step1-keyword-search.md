# Milestone 1 — limited-vocab keyword-search baseline

One linear pipeline, no forks. `data/data.jsonl` is the append-only master; every
iteration re-slices it with **`bun run regen`** and retrains into `pt/`.

## The slice

`bun run regen` defaults are set for this milestone:

| flag | value | effect |
|---|---|---|
| `--min-count` | 150 | emoji kept in `data/labels.json` iff ≥150 kept rows → **~400-emoji vocab** |
| `--max-count` | 400 | per-emoji kept-row cap → flat head/tail (ratio ≈ 10×) |
| `--n` | 4000 | eval-split size (`data/eval.jsonl`); rest → `data/train.jsonl` |

Pick these from `bun run regen --matrix` (kept-rows / vocab grid). Raising
`--min-count` shrinks the vocab; `--max-count` trades total rows for a flatter
per-emoji distribution — the flat region is what makes `text.acc@1` hold across
the tail, not just the head.

`regen` also writes the non-learned keyword vector (`kw` field on every row +
`web/public/kwproj.json`): `data/ii.json` postings projected to the vocab, each
posting list **sorted by `data/emoji_popularity.json`** and its top entry given a
`PRIMARY_BONUS` (0.15) so the canonical emoji outranks its co-listed siblings.
(`web/src/keywords.js` still needs the matching `+PRIMARY_BONUS` for browser
parity — follow-up, off the training path.)

## The model

`model/config.py` is sized small and is authoritative: `MAX_TEXT_LEN 32`,
`ENCODER_CHANNELS [64, 96]`, `ENCODER_DILATION [1, 2]` (RF 7), `TEXT_EMBED_SIZE
160`, `EMOJI_EMBED_SIZE 32`, `DROPOUT_EMOJI 0.1`, `TASK_BATCH_SIZE 128`,
`EPOCHS_TASK 1500`. Heads: `emoji` + `fusion` only.

Train and report:

```
bun run regen
train enc --local --heads emoji,fusion
uv run python tools/report.py --pt pt
```

`train enc` is stage-1 only — no GAN, no `web/public/` export.

## The goals

`goal/<date>-<sha>.yml` (newest wins; schema in `goal/README.md`). Milestone-1
gates, everything else deferred:

| goal | source | note |
|---|---|---|
| `keyword.acc@1` ≥ 0.90 | `report.json.cldr.acc_at_k[0]` | CLDR keyword probe — **model-only** today; a fused CLDR probe (needs a Python `kw`-from-text port) is the likely next step if this lags |
| `text.acc@1` ≥ 0.70 | `report.json.emoji.eval.<best fusion>_acc_at_k[0]` | short-text emoji retrieval |

`tools/report.py` grades the run against the newest goal file into
`report.json.goals`; the `planning-emojic-improvements` skill reads that, writes
the next goal file, derives `plans/<stamp>/plan.md`, implements, repeats.
