# Data files reorganization — design

## Problem

The data pipeline's file roles are inverted from what they should be. Today
`train.jsonl` (95,742 rows) is the real master that every growth tool appends
to; `eval.jsonl` (1,500 rows) is carved out of it by `split-eval.ts`;
`labels.json` is regenerated from `train.jsonl`; and `data.jsonl` is an empty,
fully-drained legacy pool. There is no single append-only source of truth, and
the "derived" files (`train.jsonl`, `eval.jsonl`, `labels.json`) all sit in git
history accumulating large diffs.

Target model:

- `data.jsonl` — the master. Append-only, never rewritten, never deleted from.
  Contains all data. Committed to git.
- `train.jsonl` — a subset of `data.jsonl`, deletable and regenerable.
- `eval.jsonl` — a subset of `data.jsonl`, deletable and regenerable.
- `labels.json` — the label vocab, deletable and regenerable from `data.jsonl`.

"Regenerable" means *reproducible from `data.jsonl` given a fixed seed* — not
byte-identical to the current files. The current hand-curated `eval.jsonl` is
not preserved; a fresh seeded split replaces it.

## Scope

In scope:

- One-time migration: merge current `train.jsonl` + `eval.jsonl` into
  `data.jsonl`.
- New `tools/data/regen.ts` — rebuilds `train.jsonl` + `eval.jsonl` +
  `labels.json` from `data.jsonl` in one command.
- Rewire growth tools (`train.ts`, `upsample-emojis.ts`,
  `upsample-emoji-test.ts`) to append to `data.jsonl`.
- Delete obsolete tools (`snapshot.ts`, `merge.ts`, `re-annotate.ts`,
  `labels.ts`, `split-eval.ts`).
- `.claude/` updates: `settings.json` PostToolUse hook, `stat.ts`,
  `/data-quality` command, `/update-model-md` command.
- `package.json` script aliases, `.gitignore`, `CLAUDE.md`.

Out of scope:

- Row schema — `{text, emojis, styles, bg, fg}` is unchanged everywhere.
- `data.py` parsing (`read`, `normalize`, `EmojiDataset`) — unchanged.
- Any training / model / export code.
- `tools/data/ekman-migrate.ts` — a historical one-off legacy migration; left
  as-is.
- Root `data.md` — stale auxiliary doc; not updated here.
- Trimming unused exports from `pool.ts` — optional, noted but not required.

## Row schema (unchanged)

Every file — master and derived — holds one JSON object per line:

```json
{"text": "...", "emojis": "😀 🎉", "styles": ["Earnest"], "bg": ["#rrggbb", "#rrggbb"], "fg": "#rrggbb"}
```

`emojis` is a single space-separated string of 0–N emojis. `styles` is a list
of 1–N labels. `bg` is two hex stops, `fg` one hex. No `meta` field is added
(the current corpus has none despite older CLAUDE.md text).

`data.jsonl` differs from the derived files in one way only: it is **raw**. The
same `normalize(text)` key may appear on multiple lines with divergent
`emojis` / `styles` / palette, because every growth run just appends. The
derived files carry exactly one row per key after `regen.ts` collapses them.

## Architecture

```
grow: train.ts / upsample-emojis.ts / upsample-emoji-test.ts
        └── append raw rows ──▶ data.jsonl  (master, committed, append-only)
                                   │
                          bun run regen
                                   │
                 ┌─────────────────┼─────────────────┐
          collapse by            seeded            top-N emoji
        normalize(text)          shuffle           frequency
          union labels          n→eval             over master
        hash-pick palette       rest→train              │
                 │                 │                     │
                 ▼                 ▼                     ▼
            (clean rows)     train.jsonl           labels.json
                             eval.jsonl            (styles = fixed set)
                             [gitignored]          [gitignored]
                                   │
                          uv run python train.py
```

## Components

### `tools/data/regen.ts` (new)

Entry point: `bun run tools/data/regen.ts` (alias `bun run regen`). Reads
`./data.jsonl` only. Writes `./train.jsonl`, `./eval.jsonl`, `./labels.json`.
Run from repo root.

Flags:

- `--seed <int>` — shuffle seed for the train/eval split. Default `42`.
- `--n <int>` — eval row count. Default `1500`.

Steps:

1. **Read** `data.jsonl` via `io.ts:readJsonl`.
2. **Collapse.** For each row, `key = normalize(text)` (import `normalize`
   from `./normalize.ts`). Skip rows whose key is empty. Accumulate per key:
   - `text` — first non-empty raw `text` seen for the key.
   - `emojis` — union (Set) of `splitEmojis(row.emojis)` across all rows for
     the key. `splitEmojis` from `./emoji.ts`.
   - `styles` — union (Set) of `row.styles` entries that are in `STYLE_SET`
     (`./styles.ts`).
   - `palettes` — every `{bg, fg}` where `bg` is a 2-element string array and
     `fg` a string.
   Emit one record per key: `{text, emojis: [...emojis].join(" "), styles:
   [...styles], bg, fg}` where `{bg, fg}` is chosen from `palettes` by
   `stableHash(key)` (import `stableHash` from `./pool.ts`), using the same
   index formula as the deleted `merge.ts:pickColor`:
   `palettes[Math.floor((((stableHash(key) * 1664525 + 1013904223) >>> 0) / 2 ** 32) * palettes.length)]`.
   A key with no valid palette is still emitted with no `bg`/`fg` (matches
   `merge.ts` `noColor` behavior; `data.py:read` will later drop it for
   lacking `bg`/`fg` — acceptable and rare).
3. **Split.** Deterministic LCG shuffle of the collapsed record array (reuse
   `split-eval.ts`'s `shuffle(rows, seed)` — LCG
   `s = (s * 1664525 + 1013904223) >>> 0`, Fisher–Yates). First `n` →
   `eval.jsonl`, remainder → `train.jsonl`.
4. **Labels.** `styles` = `[...STYLES]` from `./config` (the fixed closed
   set). `emojis` = top `TOP_EMOJIS` (`./config`, currently 352) emoji tokens
   by **row frequency over the whole collapsed master** (count each key once
   per distinct emoji it carries; sort by count desc, ties keep first-seen
   order — same as the deleted `labels.ts:topN`). Write
   `JSON.stringify({styles, emojis}, null, 2) + "\n"`.
5. **Write** all three files with `io.ts:writeFileAtomic` (no `.bak`, no
   archiving).
6. **Summary** to stdout:
   ```
   master lines read   : <N>
   distinct keys        : <K>
   keys merged (2+ rows): <M>
   -> train.jsonl       : <T>
   -> eval.jsonl        : <E>
   -> labels.json       : <S> styles, <emojis> emojis
   distinct emojis      : <D>
   ```

No `import.meta.main` guard is strictly required (single-purpose script), but
follow the repo pattern and keep the body under `if (import.meta.main)` with
pure helpers (`collapse`, `pickPalette`) exported for a future test. Add a
minimal `tools/data/regen.test.ts` covering `collapse` (union + hash palette
determinism) and the split (seeded, disjoint, sizes).

### Growth tools — rewired

| file | change |
| --- | --- |
| `tools/data/train.ts` | `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"`; `appendJsonl(DATA, lines)`; summary label `annotated -> train` → `annotated -> data`. |
| `tools/data/upsample-emojis.ts` | `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"` for both the `readJsonl` rarity count and the `appendJsonl` target. `LABELS = "./labels.json"` unchanged (present on disk after a `regen`). Summary label `appended -> train` → `appended -> data`. |
| `tools/data/upsample-emoji-test.ts` | `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"`; `appendJsonl(DATA, lines)`. Summary label updated. |

All three stay append-only and never read back what they wrote. Workflow:
grow → `bun run regen` → `uv run python train.py`.

### Deleted

- `tools/data/snapshot.ts` — built the old `data.jsonl` pool from
  `train.jsonl`; the master is now authored directly.
- `tools/data/merge.ts` — 3-file dedupe/merge that rewrote all three files;
  its collapse logic (`pickColor`, union of emojis/styles) moves into
  `regen.ts`, and rewriting `data.jsonl` violates append-only.
- `tools/data/re-annotate.ts` — drained the legacy text-only `data.jsonl`
  pool into `train.jsonl`; the pool is fully consumed and the drain semantics
  conflict with append-only.
- `tools/data/labels.ts` — absorbed into `regen.ts` step 4.
- `tools/data/split-eval.ts` — absorbed into `regen.ts` step 3.

`tools/data/pool.ts` is kept: `regen.ts` imports `stableHash`. After the
deletions its other exports (`dedupe`, `sortPool`, `freqBucket`, `MAX_TEXT_LEN`)
have no non-test caller; `pool.test.ts` keeps them covered. Trimming them is
optional and explicitly not required by this spec.

### `package.json` scripts

Remove: `snapshot`, `re-annotate`, `merge`, `split-eval`, `labels`.
Add: `"regen": "bun run tools/data/regen.ts"`.
Keep: `train`, `stat`, `upsample-emojis`, `upsample-emoji-test`, `fails`,
`preview*`, `web*`.

## One-time migration

Executed once as part of this change (a shell step in the implementation, not
a committed script):

1. `cat train.jsonl eval.jsonl > data.jsonl` — raw concat. The two files are
   currently disjoint (eval was carved out of train and removed), so the
   concat is already their union; no dedupe here — `regen.ts` owns collapse.
   Result ≈ 97,242 lines.
2. `git rm --cached train.jsonl eval.jsonl labels.json` — stop tracking the
   derived files. They remain on disk so nothing breaks between this step and
   the first `regen` run.
3. Add `train.jsonl`, `eval.jsonl`, `labels.json` to `.gitignore`.
4. `bun run regen` — overwrites the on-disk `train.jsonl` / `eval.jsonl` /
   `labels.json` with freshly derived versions.
5. Commit: `data.jsonl` (now populated, tracked), `.gitignore`, the deleted
   tool files, rewired tools, `package.json`, `.claude/` changes, `CLAUDE.md`,
   this spec.

The pre-existing tracked snapshots (`train.jsonl.pre-dedup`,
`eval.jsonl.pre-ekman`, `data.jsonl.pre-naus-drop`, etc.) are left untouched.

## Python / Modal / web consequences

Consequence of "commit `data.jsonl` only":

- `data.py`, `run.py`, `loop_emoji.py`, `test_emoji.py` — **no code change**.
  They still read `train.jsonl` / `eval.jsonl`. New precondition: `bun run
  regen` must run after every checkout or pull that changed `data.jsonl`,
  before training or evaluation. Documented in `CLAUDE.md`.
- `config.py` imports `labels.json` at module load. On a fresh checkout,
  `import config` (and therefore `data`, `train`, `export_onnx`) fails until
  `bun run regen` has produced `labels.json`. This is the sharp edge of the
  choice and gets a loud note at the top of the `CLAUDE.md` environment
  section.
- `train-modal.py` `CODE_FILES` mounts local `labels.json`, `train.jsonl`,
  `eval.jsonl`. These still exist locally after a `regen`, so the mount keeps
  working. Add `"data.jsonl"` to `CODE_FILES` for completeness.
- `web/src/feelings.test.js` reads root `labels.json` for the style-label
  list. `npm test` in `web/` now needs a prior `regen`. This is **not** in CI
  — the Pages workflow runs only `npm run build`, which consumes the committed
  `web/public/meta.json`, not root `labels.json` — so deploys are unaffected.
  A dev note goes in `CLAUDE.md`.

## `.claude/` updates

### `.claude/settings.json` — PostToolUse hook

Current: after a `Write|Edit` whose basename is `eval.jsonl` or `train.jsonl`,
run `bun run tools/data/stat.ts`. Change the `case` match from
`eval.jsonl|train.jsonl` to `data.jsonl` — the master is the file that changes
when the corpus grows, and `stat.ts` now reports it.

### `tools/data/stat.ts`

- `const FILES = ["./train.jsonl", "./eval.jsonl"]` → `const FILES =
  ["./data.jsonl"]`.
- The report now has a single file section instead of the side-by-side pair.
  Top-label coverage, style distribution, text-length histogram, emoji
  distribution — all computed over `data.jsonl` as-is (raw, pre-collapse).
- `report/data-stat/<MM-DD-HH:MM>.md` output path and section format otherwise
  unchanged.
- Header/intro text that says "train.jsonl and eval.jsonl side by side" →
  "data.jsonl (raw master)".

### `.claude/commands/data-quality.md` — `/data-quality`

Rewrite around `data.jsonl` as the primary object ("judge data.jsonl
directly"):

- **Frontmatter `description`** and intro — "Read-only data-quality report for
  `data.jsonl`" (drop "+ eval.jsonl"). Still read-only, still writes one report
  to `report/data-quality/<MM-DD-HH:MM>.md`.
- **Section 1 (distributions)** — `bun run tools/data/stat.ts` then read the
  single-file `data.jsonl` report.
- **Section 2 (structural checks)** — the embedded Python operates on
  `FILES = ["data.jsonl"]`. Keep: schema / required-key check, `normalize` →
  empty, `normalize` len > `MAX_TEXT_LEN`, no-in-set-style, raw length 4–48,
  off-closed-set styles, in-set styles per row, emoji vocab / OOV, malformed
  `bg`/`fg`. **Add**: duplicate `normalize(text)` key rate and count of keys
  that carry 2+ raw rows (the collapse surface). **Remove**: the
  `train/eval leakage` check (no split exists pre-`regen`) and the
  `with_meta` / `meta.src` / `meta.topic` provenance block (no `meta` in the
  corpus). "ROWS USED FOR TRAINING" survival count stays, computed over the
  master.
- **Section 3 (sample)** — `shuf -n 200 data.jsonl > "$STAMP.sample.jsonl"`.
  One sample file, not two.
- **Section 4 (judging)** — judge the 200-row `data.jsonl` sample; styles /
  emojis / palette keep-or-drop reasoning, unchanged in method.
- **Section 5 (report skeleton)** — collapse the "train sample / eval sample"
  pair into one "sample <n> of <N>". Drop the eval column from the schema
  table. Keep systematic-patterns and verdict sections.
- **Closing invariant check** — `git status --short` should show only the new
  files under `report/`; `data.jsonl` and `labels.json` byte-identical to
  before the run. (`train.jsonl` / `eval.jsonl` are now gitignored and
  irrelevant to this check.)

### `.claude/commands/update-model-md.md` — `/update-model-md`

- The "Dataset shape" step's `wc -l data.jsonl` / `json.loads` over
  `data.jsonl` is now correct as the full-corpus count (previously
  `data.jsonl` was the drained legacy pool). Keep it.
- Add one clause: `data.jsonl` is the raw master (may contain duplicate keys);
  the number of rows actually trained on is what `data.py:read('train.jsonl')`
  yields **after** `bun run regen`. The command should report both.
- Any reference to reading `train.jsonl` for corpus size → note it is a
  post-`regen` derived view.

## `CLAUDE.md`

Rewrite the data-pipeline portions:

- **New file-roles table** near the top of the data section: `data.jsonl`
  master / append-only / committed; `train.jsonl` + `eval.jsonl` + `labels.json`
  derived / gitignored / rebuilt by `bun run regen`.
- Replace the `train.ts` / `re-annotate.ts` / `labels.ts` / `stat.ts` bullets:
  `train.ts` (and the upsamplers) append to `data.jsonl`; `regen.ts` is the
  single derivation step; `re-annotate.ts` / `snapshot.ts` / `merge.ts` /
  `labels.ts` / `split-eval.ts` are gone.
- **Training corpus** bullet: `data.py` reads `train.jsonl` / `eval.jsonl`,
  which are produced by `bun run regen` from `data.jsonl`; run `regen` after
  any pull that touched `data.jsonl`.
- **Environment & commands**: add the precondition that `bun run regen` must
  run before `uv run python train.py` / `run.py` / `export_onnx.py` on a fresh
  checkout (they import `config`, which loads `labels.json`). Update the "data
  toolchain is Bun" line: `bun run train` / `upsample-*` grow `data.jsonl`,
  then `bun run regen`. Note `npm test` in `web/` also needs a prior `regen`.
- Update `package.json` alias list.
- Update the `stat.ts` / PostToolUse hook description (fires on `data.jsonl`,
  reports the master).

## Verification

- `bun run regen` exits 0 and writes `train.jsonl` + `eval.jsonl` +
  `labels.json`; summary counts sane (≈ 95.7k train, 1.5k eval, ~350 emojis,
  21 styles).
- Re-running `bun run regen` with the same seed produces byte-identical
  `train.jsonl` / `eval.jsonl` / `labels.json`.
- `bun test tools/data/` green (existing `pool` / `normalize` / `train` /
  `upsample` tests plus new `regen.test.ts`).
- `uv run ruff check .` and `uv run ruff format --check .` clean (no Python
  changed, should be a no-op).
- `uv run python -c "import data; rows = list(data.read('train.jsonl')); print(len(rows), rows[0])"`
  parses and prints a `record`.
- `git status` shows `data.jsonl` tracked + modified, `train.jsonl` /
  `eval.jsonl` / `labels.json` untracked (ignored), five tool files deleted.
- Not run: full `uv run python train.py` (slow; behavior of `data.py` is
  unchanged so a training run adds no signal here).
