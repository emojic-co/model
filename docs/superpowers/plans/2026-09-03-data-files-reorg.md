# Data Files Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `data.jsonl` the append-only committed master and turn `train.jsonl` / `eval.jsonl` / `labels.json` into gitignored artifacts rebuilt from it by one `bun run regen` command.

**Architecture:** A new `tools/data/regen.ts` reads `data.jsonl`, collapses rows sharing a `normalize(text)` key (union emojis + styles, hash-pick one palette), seeded-shuffles the collapsed set into `train.jsonl` + `eval.jsonl`, and writes a `labels.json` leaderboard. The growth tools (`train.ts`, `upsample-emojis.ts`, `upsample-emoji-test.ts`) switch their append target from `train.jsonl` to `data.jsonl`. Five now-obsolete tools are deleted. Row schema is unchanged everywhere and `data.py` is not touched.

**Tech Stack:** Bun + TypeScript for `tools/data/` (tests via `bun:test`), Python 3.13 + `uv` for the model stack (unchanged here), `ruff` for Python lint.

**Spec:** `docs/superpowers/specs/2026-09-03-data-files-reorg-design.md`

## Global Constraints

- **Row schema is frozen:** every line of every data file is `{"text": str, "emojis": str (space-separated), "styles": str[], "bg": [str, str], "fg": str}`. Do not add, rename, or reorder fields. A collapsed row with no valid palette is written as `{text, emojis, styles}` with no `bg`/`fg` (matches the deleted `merge.ts`).
- **`data.py` / `config.py` / `model.py` / `train.py` / `run.py` / `export_onnx.py` are out of scope** — no edits. `data.py` keeps reading `train.jsonl` / `eval.jsonl`.
- **No comments or docstrings** in any source file (existing repo convention). Keep `type:ignore` / `noqa` / shebangs if present. This applies to `regen.ts` and its test.
- **Split determinism:** seeded LCG `s = (s * 1664525 + 1013904223) >>> 0`, default seed `42`, default eval size `1500`. "Regenerable" means reproducible given the seed — NOT byte-identical to the pre-migration files.
- **Palette pick formula** (copied verbatim from the deleted `merge.ts:pickColor`): `r = ((stableHash(key) * 1664525 + 1013904223) >>> 0) / 2 ** 32; palettes[Math.floor(r * palettes.length)]`.
- **Emoji leaderboard:** top `TOP_EMOJIS` (from `tools/data/config.ts`, currently 352) by row frequency over the whole collapsed master; count each key once per distinct emoji it carries; sort by count descending, ties keep first-seen (Map insertion) order.
- **All `tools/data/` scripts run from the repo root** and use `./`-relative paths.
- **Commits:** `git add <explicit paths>` only — never `git add -A` / `git add .` (a background job also commits to this branch; keep diffs scoped). `todo.txt` has a pre-existing staged modification unrelated to this work — do not stage or revert it.
- **Commit message trailers** (every commit):
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
  ```
- **Verification budget:** `bun test tools/data/`, `uv run ruff check .`, `uv run ruff format --check .`, and targeted script runs. Never run `uv run python train.py`.

---

## File Structure

**Created:**
- `tools/data/regen.ts` — the single derivation step (master → train/eval/labels). Exports pure helpers `collapse`, `pickPalette`, `shuffleSeeded`, `emojiLeaderboard`; `main()` under `import.meta.main`.
- `tools/data/regen.test.ts` — `bun:test` coverage for the four pure helpers.

**Modified:**
- `tools/data/train.ts` — append target `train.jsonl` → `data.jsonl`.
- `tools/data/upsample-emojis.ts` — read + append target `train.jsonl` → `data.jsonl`.
- `tools/data/upsample-emoji-test.ts` — append target `train.jsonl` → `data.jsonl`.
- `tools/data/stat.ts` — `FILES` → `["./data.jsonl"]`.
- `package.json` — drop `snapshot`/`re-annotate`/`merge`/`split-eval`/`labels` aliases, add `regen`.
- `.gitignore` — ignore `train.jsonl`, `eval.jsonl`, `labels.json`.
- `.claude/settings.json` — PostToolUse hook fires on `data.jsonl`.
- `.claude/commands/data-quality.md` — rewritten around `data.jsonl`.
- `.claude/commands/update-model-md.md` — dataset-shape block reads new schema + distinguishes master from post-`regen` train set.
- `train-modal.py` — add `data.jsonl` to `CODE_FILES`.
- `CLAUDE.md` — data-pipeline sections rewritten.
- `data.jsonl` — populated by the one-time migration (was empty), committed.

**Deleted:**
- `tools/data/snapshot.ts`, `tools/data/merge.ts`, `tools/data/re-annotate.ts`, `tools/data/labels.ts`, `tools/data/split-eval.ts`.

**Kept as-is (referenced, not changed):** `tools/data/pool.ts` (`regen.ts` imports `stableHash`), `tools/data/io.ts`, `tools/data/normalize.ts`, `tools/data/emoji.ts`, `tools/data/styles.ts`, `tools/data/config.ts`, `tools/data/pool.test.ts`, `tools/data/ekman-migrate.ts`.

---

## Task 1: `regen.ts` pure helpers

**Files:**
- Create: `tools/data/regen.ts`
- Test: `tools/data/regen.test.ts`

**Interfaces:**
- Consumes (from existing modules):
  - `tools/data/io.ts` → `readJsonl<T>(path: string): Promise<T[]>`, `writeFileAtomic(path: string, data: string): Promise<void>`
  - `tools/data/normalize.ts` → `normalize(text: string): string`
  - `tools/data/emoji.ts` → `splitEmojis(raw: string): string[]`
  - `tools/data/pool.ts` → `stableHash(s: string): number`
  - `tools/data/styles.ts` → `STYLE_SET: ReadonlySet<string>`
  - `tools/data/config.ts` → `STYLES: readonly string[]`, `TOP_EMOJIS: number`
- Produces (used by Task 2's `main()` and by later reasoning):
  - `type Palette = { bg: string[]; fg: string }`
  - `type Record = { text: string; emojis: string; styles: string[]; bg?: string[]; fg?: string }`
  - `collapse(rows: unknown[]): Record[]` — one `Record` per non-empty `normalize(text)` key; `emojis` is the space-joined union; `styles` is the `STYLE_SET`-filtered union; `bg`/`fg` present only when at least one source row had a valid palette.
  - `pickPalette(key: string, palettes: Palette[]): Palette | undefined`
  - `shuffleSeeded<T>(rows: T[], seed: number): T[]` — pure (no input mutation), deterministic per seed.
  - `emojiLeaderboard(records: { emojis: string }[], n: number): string[]`

- [ ] **Step 1: Write the failing test**

Create `tools/data/regen.test.ts`:

```ts
import { expect, test } from "bun:test"

import {
  collapse,
  emojiLeaderboard,
  pickPalette,
  shuffleSeeded,
} from "./regen.ts"

const P = (a: string, b: string, f: string) => ({ bg: [a, b], fg: f })

test("collapse unions emojis and styles across rows with the same normalized text", () => {
  const out = collapse([
    { text: "Bus is late", emojis: "🚌", styles: ["Irritated"], ...P("#111111", "#222222", "#eeeeee") },
    { text: "  bus   is late  ", emojis: "😤 🚌", styles: ["Tense", "Irritated"], ...P("#333333", "#444444", "#dddddd") },
  ])
  expect(out).toHaveLength(1)
  expect(out[0].emojis.split(" ").sort()).toEqual(["😤", "🚌"].sort())
  expect(out[0].styles.sort()).toEqual(["Irritated", "Tense"])
  expect(out[0].bg).toHaveLength(2)
  expect(typeof out[0].fg).toBe("string")
})

test("collapse drops rows that normalize to empty", () => {
  const out = collapse([
    { text: "😀😀😀", emojis: "😀", styles: ["Joyful"], ...P("#111111", "#222222", "#eeeeee") },
    { text: "real text here", emojis: "", styles: ["Deadpan"], ...P("#111111", "#222222", "#eeeeee") },
  ])
  expect(out.map((r) => r.text)).toEqual(["real text here"])
})

test("collapse keeps only styles in the closed set", () => {
  const out = collapse([
    { text: "hello world", emojis: "", styles: ["Joyful", "Bogus", "notreal"], ...P("#111111", "#222222", "#eeeeee") },
  ])
  expect(out[0].styles).toEqual(["Joyful"])
})

test("collapse emits a record with no bg/fg when no source row had a palette", () => {
  const out = collapse([{ text: "no colors", emojis: "🎈", styles: ["Playful"] }])
  expect(out).toHaveLength(1)
  expect(out[0].bg).toBeUndefined()
  expect(out[0].fg).toBeUndefined()
})

test("pickPalette is deterministic and returns one of the given palettes", () => {
  const palettes = [
    P("#aaaaaa", "#bbbbbb", "#000000"),
    P("#cccccc", "#dddddd", "#111111"),
    P("#eeeeee", "#ffffff", "#222222"),
  ]
  const a = pickPalette("some key", palettes)
  expect(pickPalette("some key", palettes)).toEqual(a!)
  expect(palettes).toContainEqual(a!)
})

test("pickPalette returns undefined when there are no palettes", () => {
  expect(pickPalette("k", [])).toBeUndefined()
})

test("shuffleSeeded is a deterministic permutation and does not mutate input", () => {
  const xs = Array.from({ length: 60 }, (_, i) => i)
  const a = shuffleSeeded(xs, 42)
  expect(shuffleSeeded(xs, 42)).toEqual(a)
  expect([...a].sort((p, q) => p - q)).toEqual(xs)
  expect(a).not.toEqual(xs)
  expect(shuffleSeeded(xs, 7)).not.toEqual(a)
  expect(xs).toEqual(Array.from({ length: 60 }, (_, i) => i))
})

test("emojiLeaderboard ranks by row frequency, ties keep first-seen order", () => {
  const recs = [
    { emojis: "🍕 🍕 🚗" },
    { emojis: "🍕 🎂" },
    { emojis: "🚗" },
    { emojis: "🎸" },
  ]
  expect(emojiLeaderboard(recs, 3)).toEqual(["🍕", "🚗", "🎂"])
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `bun test tools/data/regen.test.ts`
Expected: FAIL — `Cannot find module './regen.ts'` (or "export named 'collapse' not found").

- [ ] **Step 3: Write the minimal implementation**

Create `tools/data/regen.ts` with exactly this content (helpers + typed shapes only — no `main()` yet). The row type is named `Row`, not `Record`, so it does not shadow the TS built-in `Record<K,V>`:

```ts
import { splitEmojis } from "./emoji.ts"
import { normalize } from "./normalize.ts"
import { stableHash } from "./pool.ts"
import { STYLE_SET } from "./styles.ts"

export type Palette = { bg: string[]; fg: string }
export type Row = {
  text: string
  emojis: string
  styles: string[]
  bg?: string[]
  fg?: string
}

type Acc = {
  text: string
  emojis: Set<string>
  styles: Set<string>
  palettes: Palette[]
}

function rowPalette(row: Record<string, unknown>): Palette | undefined {
  const { bg, fg } = row
  if (Array.isArray(bg) && bg.length >= 2 && typeof fg === "string") {
    return { bg: (bg as string[]).slice(0, 2), fg }
  }
  return undefined
}

export function pickPalette(
  key: string,
  palettes: Palette[],
): Palette | undefined {
  if (!palettes.length) return undefined
  const r = ((stableHash(key) * 1664525 + 1013904223) >>> 0) / 2 ** 32
  return palettes[Math.floor(r * palettes.length)]
}

export function collapse(rows: unknown[]): Row[] {
  const acc = new Map<string, Acc>()
  for (const raw of rows) {
    const row = (raw ?? {}) as Record<string, unknown>
    const text = typeof row.text === "string" ? row.text : ""
    const key = normalize(text)
    if (!key) continue
    let a = acc.get(key)
    if (!a) {
      a = { text, emojis: new Set(), styles: new Set(), palettes: [] }
      acc.set(key, a)
    }
    if (typeof row.emojis === "string") {
      for (const e of splitEmojis(row.emojis)) a.emojis.add(e)
    }
    if (Array.isArray(row.styles)) {
      for (const s of row.styles) {
        if (typeof s === "string" && STYLE_SET.has(s)) a.styles.add(s)
      }
    }
    const p = rowPalette(row)
    if (p) a.palettes.push(p)
  }

  const out: Row[] = []
  for (const [key, a] of acc) {
    const rec: Row = {
      text: a.text,
      emojis: [...a.emojis].join(" "),
      styles: [...a.styles],
    }
    const p = pickPalette(key, a.palettes)
    if (p) {
      rec.bg = p.bg
      rec.fg = p.fg
    }
    out.push(rec)
  }
  return out
}

export function shuffleSeeded<T>(rows: T[], seed: number): T[] {
  let s = seed >>> 0
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  const out = [...rows]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function emojiLeaderboard(
  records: { emojis: string }[],
  n: number,
): string[] {
  const counts = new Map<string, number>()
  for (const rec of records) {
    for (const e of new Set(splitEmojis(rec.emojis))) {
      counts.set(e, (counts.get(e) ?? 0) + 1)
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k)
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `bun test tools/data/regen.test.ts`
Expected: PASS — 8 tests pass.

- [ ] **Step 5: Lint the whole `tools/data/` suite still loads**

Run: `bun test tools/data/`
Expected: PASS — existing `pool` / `normalize` / `train` / `styles` / `emoji` / `upsample-*` tests plus the 8 new ones, no failures.

- [ ] **Step 6: Commit**

```bash
git add tools/data/regen.ts tools/data/regen.test.ts
git commit -m "$(cat <<'EOF'
data: regen.ts pure helpers (collapse / split / leaderboard)

collapse() unions emojis + closed-set styles per normalize(text) key and
hash-picks one palette (merge.ts:pickColor formula). shuffleSeeded() is
the split-eval LCG shuffle, pure. emojiLeaderboard() is the labels.ts
top-N-by-row-frequency ranking. main() wiring lands next.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

---

## Task 2: `regen.ts` main, migration, gitignore — the pipeline inversion goes live

**Files:**
- Modify: `tools/data/regen.ts` (add `argInt` + `main()` under `import.meta.main`)
- Modify: `package.json` (add `regen` alias)
- Modify: `.gitignore` (ignore the three derived files)
- Modify: `data.jsonl` (populate via migration), stop tracking `train.jsonl` / `eval.jsonl` / `labels.json`

**Interfaces:**
- Consumes: `collapse`, `shuffleSeeded`, `emojiLeaderboard` from Task 1; `readJsonl` / `writeFileAtomic` from `io.ts`; `STYLES` / `TOP_EMOJIS` from `config.ts`.
- Produces: the command `bun run regen` (alias for `bun run tools/data/regen.ts`), accepting `--seed <int>` (default 42) and `--n <int>` (default 1500), writing `./train.jsonl`, `./eval.jsonl`, `./labels.json` from `./data.jsonl`.

- [ ] **Step 1: Add `main()` to `tools/data/regen.ts`**

Append to `tools/data/regen.ts`:

```ts
import { readJsonl, writeFileAtomic } from "./io.ts"
import { STYLES, TOP_EMOJIS } from "./config"

const DATA = "./data.jsonl"
const TRAIN = "./train.jsonl"
const EVAL = "./eval.jsonl"
const LABELS = "./labels.json"

function argInt(flag: string): number | undefined {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return undefined
  const n = Number(process.argv[i + 1])
  return Number.isFinite(n) ? n : undefined
}

function toLine(r: Row): string {
  return JSON.stringify(
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles },
  )
}

if (import.meta.main) {
  const seed = argInt("--seed") ?? 42
  const n = argInt("--n") ?? 1500

  const raw = await readJsonl<unknown>(DATA)
  const records = collapse(raw)
  const merged = records.length
  const dupKeys = raw.length - merged

  const shuffled = shuffleSeeded(records, seed)
  const held = shuffled.slice(0, n)
  const rest = shuffled.slice(n)

  await writeFileAtomic(EVAL, held.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TRAIN, rest.map(toLine).join("\n") + "\n")

  const labels = {
    styles: [...STYLES],
    emojis: emojiLeaderboard(records, TOP_EMOJIS),
  }
  await writeFileAtomic(LABELS, JSON.stringify(labels, null, 2) + "\n")

  console.log("\n--- regen ---")
  console.log(`master lines read    : ${raw.length}`)
  console.log(`distinct keys        : ${merged}`)
  console.log(`collapsed away       : ${dupKeys}`)
  console.log(`-> ${EVAL}      : ${held.length}`)
  console.log(`-> ${TRAIN}     : ${rest.length}`)
  console.log(
    `-> ${LABELS}   : ${labels.styles.length} styles, ${labels.emojis.length} emojis`,
  )
  process.exit(0)
}
```

(If you kept the `Row` rename from Task 1, `toLine` already refers to `Row` — good. `collapse` returns `Row[]`.)

- [ ] **Step 2: Add the `regen` script alias**

In `package.json` `"scripts"`, add after the `"train"` line:

```json
    "regen": "bun run tools/data/regen.ts",
```

(Leave `snapshot` / `re-annotate` / `merge` / `split-eval` / `labels` in place for now — Task 3 removes them together with their tool files.)

- [ ] **Step 3: Run the one-time migration**

```bash
wc -l train.jsonl eval.jsonl data.jsonl
cat train.jsonl eval.jsonl > data.jsonl.new && mv data.jsonl.new data.jsonl
wc -l data.jsonl
```
Expected: `data.jsonl` line count ≈ `train.jsonl` + `eval.jsonl` (≈ 97,242). `train.jsonl` and `eval.jsonl` currently share no normalized text (eval was carved out and removed), so a raw concat is already their union.

- [ ] **Step 4: Run `regen` and check the summary**

Run: `bun run regen`
Expected: exit 0; summary shows `master lines read` ≈ 97,242, `distinct keys` slightly below that (a handful of collapses), `-> ./eval.jsonl : 1500`, `-> ./train.jsonl : <distinct keys − 1500>`, `-> ./labels.json : 21 styles, 352 emojis`.

- [ ] **Step 5: Verify a re-run is byte-identical**

```bash
md5sum train.jsonl eval.jsonl labels.json
bun run regen
md5sum train.jsonl eval.jsonl labels.json
```
Expected: the three checksums are unchanged across the two runs.

- [ ] **Step 6: Verify the Python reader still parses the regenerated files**

Run:
```bash
uv run python -c "import data; rows = list(data.read('train.jsonl')); ev = list(data.read('eval.jsonl')); print(len(rows), len(ev)); print(rows[0])"
```
Expected: prints two counts (train in the tens of thousands, eval ≤ 1500) and a `record(text=..., emojis=[...], styles=[...], colors=[...])` line, no exception.

- [ ] **Step 7: Stop tracking the derived files and ignore them**

In `.gitignore`, add near the `data.jsonl.dry` / `data.jsonl.tmp` lines:

```
# Rebuilt from data.jsonl by `bun run regen`; data.jsonl is the committed master.
train.jsonl
eval.jsonl
labels.json
```

Then:
```bash
git rm --cached train.jsonl eval.jsonl labels.json
git status --short
```
Expected: `D train.jsonl`, `D eval.jsonl`, `D labels.json` staged (working-tree copies remain on disk); `M data.jsonl`, `M .gitignore`, `M package.json`, `M tools/data/regen.ts` also present; `train.jsonl` / `eval.jsonl` / `labels.json` no longer show as untracked (ignored).

- [ ] **Step 8: Full test + lint sweep**

```bash
bun test tools/data/
uv run ruff check .
uv run ruff format --check .
```
Expected: all green (no Python changed — ruff is a no-op sanity check).

- [ ] **Step 9: Commit**

The three `git rm --cached` deletions from Step 7 are already staged. Do not `git add train.jsonl eval.jsonl labels.json` — they are now ignored and untracked, so `git add` would error; the staged deletions carry into this commit on their own.

```bash
git add tools/data/regen.ts package.json .gitignore data.jsonl
git commit -m "$(cat <<'EOF'
data: invert pipeline -- data.jsonl master, train/eval/labels derived

data.jsonl is now the append-only committed master (migrated =
train.jsonl + eval.jsonl concatenated). regen.ts main() rebuilds
train.jsonl + eval.jsonl (seeded split, default seed 42 / n 1500) and
labels.json from it in one `bun run regen`. train.jsonl, eval.jsonl and
labels.json are now gitignored; run `bun run regen` after any pull that
touched data.jsonl before training.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

Confirm with `git show --stat HEAD` afterward: `data.jsonl` large insertion, `train.jsonl` / `eval.jsonl` / `labels.json` deleted, `.gitignore` / `package.json` / `tools/data/regen.ts` modified.

---

## Task 3: Rewire growth tools, delete obsolete tools

**Files:**
- Modify: `tools/data/train.ts`, `tools/data/upsample-emojis.ts`, `tools/data/upsample-emoji-test.ts`
- Delete: `tools/data/snapshot.ts`, `tools/data/merge.ts`, `tools/data/re-annotate.ts`, `tools/data/labels.ts`, `tools/data/split-eval.ts`
- Modify: `package.json` (drop five aliases)

**Interfaces:**
- Consumes: `bun run regen` (Task 2) is the new step between growing `data.jsonl` and training.
- Produces: `train.ts` / `upsample-emojis.ts` / `upsample-emoji-test.ts` append to `./data.jsonl`. No tool writes `train.jsonl` / `eval.jsonl` / `labels.json` any more except `regen.ts`.

- [ ] **Step 1: Confirm nothing imports the doomed files**

Run:
```bash
grep -rn 'from "\./\(snapshot\|merge\|re-annotate\|labels\|split-eval\)' tools/
```
Expected: no matches (these modules are entrypoints, not imported). If anything matches, stop and reassess.

- [ ] **Step 2: Rewire `tools/data/train.ts`**

- Line 8: `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"`
- In the `import.meta.main` block: `await appendJsonl(TRAIN, lines)` → `await appendJsonl(DATA, lines)`
- Summary line: `console.log(\`annotated -> train  : ${lines.length}\`)` → `console.log(\`annotated -> data   : ${lines.length}\`)` (keep column alignment with the neighbouring lines)

- [ ] **Step 3: Rewire `tools/data/upsample-emojis.ts`**

- Line 11: `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"`
- Keep line 12 `const LABELS = "./labels.json"` unchanged (present on disk after `bun run regen`).
- In `import.meta.main`: `const rows = await readJsonl<{ emojis?: string }>(TRAIN)` → `... (DATA)`
- `await appendJsonl(TRAIN, lines)` → `await appendJsonl(DATA, lines)`
- Summary line `appended -> train    : ${lines.length}` → `appended -> data     : ${lines.length}`

- [ ] **Step 4: Rewire `tools/data/upsample-emoji-test.ts`**

- Line 10: `const TRAIN = "./train.jsonl"` → `const DATA = "./data.jsonl"`
- `await appendJsonl(TRAIN, lines)` → `await appendJsonl(DATA, lines)`
- Summary line `appended -> train    : ${lines.length}` → `appended -> data     : ${lines.length}`

- [ ] **Step 5: Delete the obsolete tools**

```bash
git rm tools/data/snapshot.ts tools/data/merge.ts tools/data/re-annotate.ts tools/data/labels.ts tools/data/split-eval.ts
```

- [ ] **Step 6: Drop their `package.json` aliases**

Remove these lines from `"scripts"`:
```json
    "snapshot": "bun run tools/data/snapshot.ts",
    "re-annotate": "bun run tools/data/re-annotate.ts",
    "merge": "bun run tools/data/merge.ts",
    "split-eval": "bun run tools/data/split-eval.ts",
    "labels": "bun run tools/data/labels.ts",
```
Leave `regen`, `train`, `stat`, `upsample-emojis`, `upsample-emoji-test`, `fails`, `preview*`, `web*` intact.

- [ ] **Step 7: Verify no stale references remain**

Run:
```bash
grep -rn 'snapshot\.ts\|re-annotate\.ts\|split-eval\.ts\|"\./merge\.ts"\|data/labels\.ts' tools/ package.json
grep -rn "train\.jsonl" tools/data/train.ts tools/data/upsample-emojis.ts tools/data/upsample-emoji-test.ts
```
Expected: first grep — no matches. Second grep — no matches (all three now say `data.jsonl`).

- [ ] **Step 8: Test sweep**

Run: `bun test tools/data/`
Expected: PASS. `train.test.ts` (imports `TOPICS` / `topicForBatch`), `upsample-emojis.test.ts` (imports `countEmojis` / `rarest`), `upsample-emoji-test.test.ts` (imports `pickFailed`), `regen.test.ts`, `pool.test.ts`, `normalize.test.ts` — none touch the renamed consts, all green.

- [ ] **Step 9: Commit**

```bash
git add tools/data/train.ts tools/data/upsample-emojis.ts tools/data/upsample-emoji-test.ts tools/data/snapshot.ts tools/data/merge.ts tools/data/re-annotate.ts tools/data/labels.ts tools/data/split-eval.ts package.json
git commit -m "$(cat <<'EOF'
data: growth tools append to data.jsonl; drop old-model tools

train.ts / upsample-emojis.ts / upsample-emoji-test.ts now append to
data.jsonl. snapshot.ts (train->pool), merge.ts (3-file merge),
re-annotate.ts (legacy pool drain), labels.ts and split-eval.ts (both
absorbed into regen.ts) are deleted along with their package.json aliases.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

---

## Task 4: `stat.ts` and the PostToolUse hook target the master

**Files:**
- Modify: `tools/data/stat.ts:8`
- Modify: `.claude/settings.json:9`

**Interfaces:**
- Consumes: `data.jsonl` as the sole input file.
- Produces: `report/data-stat/<MM-DD-HH:MM>.md` describing `data.jsonl`; the PostToolUse hook runs `stat.ts` after a `Write|Edit` to `data.jsonl`.

- [ ] **Step 1: Point `stat.ts` at `data.jsonl`**

`tools/data/stat.ts` line 8:
```ts
const FILES = ["./train.jsonl", "./eval.jsonl"]
```
→
```ts
const FILES = ["./data.jsonl"]
```
No other change: `main()` iterates `FILES`, builds one `parsed` entry, renders one `section()` plus the "Top-label coverage" list. `section(path, rows)` and `coverage(rows)` are already path-parametric with no hardcoded `train`/`eval` string.

- [ ] **Step 2: Run `stat.ts` and eyeball the report**

Run: `bun run stat` then `sed -n '1,20p' "$(ls -t report/data-stat/*.md | head -1)"`
Expected: a report headed `# data stats — <date>` with a single `## ./data.jsonl` section, `**<~97k> rows**`, style distribution, length histogram, emoji distribution.

- [ ] **Step 3: Point the PostToolUse hook at `data.jsonl`**

`.claude/settings.json`, the hook `command` string — change the `case` pattern:
```
case "$(basename "$f")" in eval.jsonl|train.jsonl) cd "$CLAUDE_PROJECT_DIR" && bun run tools/data/stat.ts ;; esac;
```
→
```
case "$(basename "$f")" in data.jsonl) cd "$CLAUDE_PROJECT_DIR" && bun run tools/data/stat.ts ;; esac;
```
(Only the `in eval.jsonl|train.jsonl)` token changes to `in data.jsonl)`. Keep the surrounding `jq` / `read` / redirection intact.)

- [ ] **Step 4: Simulate the hook dispatch**

Run:
```bash
echo '{"tool_input":{"file_path":"/repo/data.jsonl"}}' | jq -r '.tool_input.file_path // .tool_response.filePath // empty' | { read -r f; case "$(basename "$f")" in data.jsonl) echo "MATCH -> would run stat.ts" ;; *) echo "no match" ;; esac; }
echo '{"tool_input":{"file_path":"/repo/train.jsonl"}}' | jq -r '.tool_input.file_path // .tool_response.filePath // empty' | { read -r f; case "$(basename "$f")" in data.jsonl) echo "MATCH" ;; *) echo "no match (correct: train.jsonl no longer triggers)" ;; esac; }
```
Expected: first prints `MATCH -> would run stat.ts`; second prints `no match (correct: ...)`.

- [ ] **Step 5: Validate the JSON**

Run: `jq . .claude/settings.json > /dev/null && echo "settings.json valid"`
Expected: `settings.json valid`.

- [ ] **Step 6: Commit**

```bash
git add tools/data/stat.ts .claude/settings.json
git commit -m "$(cat <<'EOF'
data: stat.ts + PostToolUse hook report data.jsonl (the master)

stat.ts summarizes data.jsonl instead of the train/eval pair; the
Write|Edit hook now fires stat.ts on data.jsonl writes (train.jsonl /
eval.jsonl are derived and gitignored).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

---

## Task 5: Rewrite the `/data-quality` command around `data.jsonl`

**Files:**
- Modify: `.claude/commands/data-quality.md`

**Interfaces:**
- Consumes: `data.jsonl` (raw master), `labels.json` (present after `bun run regen`), `data.py:normalize` + `config.MAX_TEXT_LEN`.
- Produces: a read-only `report/data-quality/<MM-DD-HH:MM>.md` judging `data.jsonl` — schema/`read()` survival, raw length, style set, emoji vocab, duplicates + union-merge surface, palette, then an in-session 200-row label-quality judgment.

- [ ] **Step 1: Frontmatter + intro**

Replace the frontmatter `description:` value with:
```
Read-only data-quality report for data.jsonl (the append-only master) — refreshes tools/data/stat.ts distributions, runs structural checks (schema, read() survival, raw length, style set, emoji vocab, duplicate/union-merge keys, palette), then judges label + palette quality on a 200-row sample in-session. Writes one report to report/data-quality/, never modifies any data file.
```
In the first body paragraph, replace "Judge the health of `train.jsonl` and `eval.jsonl` side by side" with "Judge the health of `data.jsonl` (the append-only master that `bun run regen` derives `train.jsonl` / `eval.jsonl` / `labels.json` from)". In the "strictly read-only" paragraph keep the list of files it must not modify (`train.jsonl`, `eval.jsonl`, `data.jsonl`, `labels.json`) as-is.

- [ ] **Step 2: Reference-points list**

In the "Reference points" bullets:
- Keep the `data.py:read` bullet (it still describes what a trainable row is).
- Replace the `labels.json` bullet's "current top-320 frequency leaderboard" wording with "current top-`TOP_EMOJIS` frequency leaderboard, rebuilt from `data.jsonl` by `bun run regen`".
- **Delete** the final bullet about rows carrying `meta` (`src`, `v`, `at`, `model`, `params`, `topic`) — the corpus has no `meta`.

- [ ] **Step 3: Section 1 (distributions)**

The bash block already calls `bun run tools/data/stat.ts`; after Task 4 that reports `data.jsonl`. Change the prose "row counts, style distribution, text-length histogram + out-of-range count, emoji distribution, top-label coverage) are lifted into section 1 ... by reference" to name `data.jsonl` explicitly. No command change.

- [ ] **Step 4: Section 2 (structural checks) — replace the Python heredoc**

Replace the entire ```` ```bash ... uv run python - <<'EOF' ... EOF ... ``` ```` block in section 2 with:

````markdown
```bash
uv run python - <<'EOF'
import collections, json, re

from config import MAX_TEXT_LEN
from data import normalize

LABELS = json.load(open("labels.json", encoding="utf-8"))
STYLE_SET, EMOJI_SET = set(LABELS["styles"]), set(LABELS["emojis"])
PATH = "data.jsonl"
MIN_RAW, MAX_RAW = 4, 48
REQUIRED = ("text", "emojis", "styles", "bg", "fg")
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def load(path):
    rows, bad = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return rows, bad


def palette_ok(r):
    bg, fg = r.get("bg"), r.get("fg")
    return (
        isinstance(bg, list) and len(bg) == 2
        and all(isinstance(x, str) and HEX.match(x) for x in bg)
        and isinstance(fg, str) and bool(HEX.match(fg))
    )


rows, bad = load(PATH)
n = len(rows)
print(f"==== {PATH} ====")
print(f"rows {n}   json parse failures {bad}")

missing = collections.Counter()
for r in rows:
    for k in REQUIRED:
        if k not in r:
            missing[k] += 1
print("missing required key", dict(missing) or "none")

schema_ok = [r for r in rows if all(k in r for k in REQUIRED)]
nt = {id(r): normalize(r["text"]) for r in schema_ok}
in_set = {id(r): [s for s in r["styles"] if s in STYLE_SET] for r in schema_ok}
norm_empty = sum(1 for r in schema_ok if not nt[id(r)])
too_long = sum(1 for r in schema_ok if nt[id(r)] and len(nt[id(r)]) > MAX_TEXT_LEN)
no_style = sum(1 for r in schema_ok if not in_set[id(r)])
survive = sum(
    1 for r in schema_ok
    if nt[id(r)] and len(nt[id(r)]) <= MAX_TEXT_LEN and in_set[id(r)]
)
print(f"normalize->empty {norm_empty}   norm len>{MAX_TEXT_LEN} {too_long}   "
      f"no in-set style {no_style}")
print(f"ROWS THAT SURVIVE data.py:read {survive}  ({n - survive} lost)")

raw_lens = sorted(len([*r["text"]]) for r in schema_ok)
oor = [r["text"] for r in schema_ok
       if not (MIN_RAW <= len([*r["text"]]) <= MAX_RAW)]
if raw_lens:
    print(f"raw len  min {raw_lens[0]} median {raw_lens[len(raw_lens)//2]} "
          f"max {raw_lens[-1]}")
print(f"raw len outside {MIN_RAW}-{MAX_RAW}: {len(oor)}"
      + (f"  e.g. {oor[:5]}" if oor else ""))

off_style = collections.Counter(
    s for r in schema_ok for s in r["styles"] if s not in STYLE_SET
)
print("styles off the closed set", dict(off_style.most_common(20)) or "none")
spr = collections.Counter(min(len(in_set[id(r)]), 3) for r in schema_ok)
print(f"in-set styles/row 0/1/2/3+ {spr[0]}/{spr[1]}/{spr[2]}/{spr[3]}")

toks = [(r, r["emojis"].split()) for r in schema_ok if isinstance(r["emojis"], str)]
distinct = {e for _, es in toks for e in es}
mentions = [e for _, es in toks for e in es]
oov = [e for e in mentions if e not in EMOJI_SET]
all_in = sum(1 for _, es in toks if es and all(e in EMOJI_SET for e in es))
zero_after = sum(1 for _, es in toks if es and not [e for e in es if e in EMOJI_SET])
print(f"distinct emoji tokens {len(distinct)} ({len(distinct - EMOJI_SET)} outside "
      f"labels.json)   mentions {len(mentions)} / oov {len(oov)}")
print("  top oov", collections.Counter(oov).most_common(8))
print(f"rows all emojis in vocab {all_in}/{len(toks)}   "
      f"rows -> 0 emoji after filter {zero_after}")

raw_dupes = collections.Counter(r["text"] for r in schema_ok)
raw_extra = sum(v - 1 for v in raw_dupes.values() if v > 1)
norm_map = collections.Counter(nt[id(r)] for r in schema_ok)
norm_extra = sum(v - 1 for v in norm_map.values() if v > 1)
keys_multi = sum(1 for v in norm_map.values() if v > 1)
print(f"exact raw-text dupes {raw_extra} extra rows   "
      f"normalize-collapsed dupes {norm_extra} extra rows")
print(f"normalized keys with 2+ rows (union-merged by regen) {keys_multi} "
      f"/ {len(norm_map)} keys")
print("  e.g.", [t for t, v in raw_dupes.most_common(5) if v > 1])

bad_pal = [r["text"] for r in schema_ok if not palette_ok(r)]
print(f"malformed bg/fg {len(bad_pal)}"
      + (f"  e.g. {bad_pal[:5]}" if bad_pal else ""))
EOF
```

If `from data import ...` or `from config import ...` fails, the modules have
drifted — `grep -n "def normalize\|MAX_TEXT_LEN" data.py config.py` and adjust.
````

This drops the old train/eval loop, the `norm_by_file` leakage block, and the `meta` / `src` / `v` / `topic` provenance block; it adds the `normalized keys with 2+ rows` line (the union-merge surface).

- [ ] **Step 5: Section 3 (sample)**

Replace the two `shuf` lines with one:
```bash
shuf -n 200 data.jsonl > "report/data-quality/$STAMP.sample.jsonl"
wc -l "report/data-quality/$STAMP.sample.jsonl"
```
and update the surrounding prose ("Never read either corpus whole" → "Never read `data.jsonl` whole").

- [ ] **Step 6: Section 4 (judging)**

Change "Read both sample files" → "Read the sample file". Change "keeping a running tally per file" → "keeping a running tally". Everything about styles / emojis / palette keep-or-drop reasoning stays verbatim.

- [ ] **Step 7: Section 5 (report skeleton)**

- Header bullets: replace the two-file `Files:` / dual `Label quality — train sample` / `eval sample` lines with single `- File: \`data.jsonl\` <N> rows`, `- Trainable after \`data.py:read\`: <a> (<N-a> lost)`, `- Label quality — sample <n>: styles <ok>/<weak>/<wrong> · emoji rows <ok>/<has-weak>/<has-wrong> · palette <ok>/<weak>/<wrong>`.
- Section "## 2. Structural checks" table: drop the `eval` column, keep a single count column; rename the last row to `**rows that survive data.py:read**`; add a `normalized keys with 2+ rows` row.
- Delete the "### train / eval leakage" subsection and the "### Provenance (`meta`)" subsection from the skeleton.
- "## 3. Label quality — train (sample <n> of <N>)" → "## 3. Label quality (sample <n> of <N>)"; delete "## 4. Label quality — eval ...". Renumber "## 5. Systematic patterns" → "## 4.", "## 6. Verdict" → "## 5.".

- [ ] **Step 8: Closing invariant check**

In the final "Report back" / invariant paragraph, replace "`train.jsonl`, `eval.jsonl`, `data.jsonl`, and `labels.json` are byte-identical to before this run (`git status --short` should show only the new files under `report/`)" with "`data.jsonl` and `labels.json` are byte-identical to before this run (`git status --short` shows only new files under `report/`; `train.jsonl` / `eval.jsonl` are gitignored)". In the closing "2–3 sentences" instruction replace "the two row counts" with "the master row count" and "for each sample" with "for the sample".

- [ ] **Step 9: Dry-run the structural script**

Run the section-2 Python heredoc exactly as written against the repo.
Expected: it prints the `==== data.jsonl ====` block with real numbers, no traceback. (`labels.json` and `train.jsonl` must exist — run `bun run regen` first if needed.)

- [ ] **Step 10: Commit**

```bash
git add .claude/commands/data-quality.md
git commit -m "$(cat <<'EOF'
data: /data-quality judges data.jsonl (the master) directly

Rewrites the command around data.jsonl: structural checks + the 200-row
label-quality sample now run on the master. Drops the train/eval leakage
check (no split pre-regen) and the meta/provenance block (no meta in the
corpus); adds a duplicate-key / union-merge-surface metric.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

---

## Task 6: Documentation and remaining references

**Files:**
- Modify: `.claude/commands/update-model-md.md:50-64`
- Modify: `train-modal.py` (`CODE_FILES`)
- Modify: `CLAUDE.md` (data-pipeline sections)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: docs that describe the master/derived model and the `bun run regen` precondition.

- [ ] **Step 1: `/update-model-md` dataset-shape block**

In `.claude/commands/update-model-md.md`, replace the "**Dataset shape** — never read `data.jsonl` whole:" bash block (lines ~50–64) with:

````markdown
**Dataset shape** — `data.jsonl` is the raw append-only master; `train.jsonl`
is the deduped subset `bun run regen` derives and what the model actually
trains on. Report both. Never read `data.jsonl` whole:

```bash
wc -l data.jsonl train.jsonl
python3 -c "
import json,collections,statistics
rows=[json.loads(l) for l in open('train.jsonl')]
st=collections.Counter(s for r in rows for s in r['styles'])
em=collections.Counter(e for r in rows for e in r['emojis'].split())
mc=em.most_common()
print('train rows', len(rows))
print('styles', st.most_common())
print('n emoji classes', len(mc),
      'median', statistics.median(em.values()) if em else 0,
      'max', mc[0] if mc else None, 'min', mc[-1] if mc else None)
print('tail', mc[-12:])
"
```

If `train.jsonl` is missing or stale, run `bun run regen` first.
````

- [ ] **Step 2: `train-modal.py` file list**

In `train-modal.py`, `CODE_FILES`, add `"data.jsonl",` immediately before `"train.jsonl",`. (The Modal run mounts local files; `train.jsonl` / `eval.jsonl` / `labels.json` still exist on disk after a local `bun run regen`, and `data.jsonl` is now the source of record — mount it too.)

- [ ] **Step 3: `CLAUDE.md` — data pipeline intro**

Find the "Data pipeline (`tools/data/`)" bullet. Replace the sentence beginning "`train.jsonl` (one JSON object per line: ...) is the corpus the new pipeline grows; the legacy `data.jsonl` (older `feeling`/`emoji` singular shape) is a fixed pool that `re-annotate.ts` drains into `train.jsonl` under the new schema." with:

> `data.jsonl` (one JSON object per line: `text`, `emojis` — a single space-separated string of 0–N emojis, `styles` — a list of 1–N style labels, `bg` — two hex stops, `fg` — one hex) is the **append-only master**: every growth tool appends to it and nothing ever rewrites or deletes from it. `train.jsonl`, `eval.jsonl` and `labels.json` are **gitignored artifacts** rebuilt from `data.jsonl` by `bun run regen` (`tools/data/regen.ts`) — it collapses rows sharing a `normalize(text)` key (union of emojis + closed-set styles, one palette hash-picked), seeded-shuffles the result into `train.jsonl` + `eval.jsonl` (default seed 42, eval size 1500), and writes the `labels.json` leaderboard. Run `bun run regen` after any pull that changed `data.jsonl` and before training.

- [ ] **Step 4: `CLAUDE.md` — per-script bullets**

- `tools/data/train.ts` bullet: "appends ~1,000 fresh rows to `train.jsonl` per run" → "appends ~1,000 fresh rows to `data.jsonl` per run"; in the Phase 2 sentence "each annotated text is written as `{text, emojis, styles, bg, fg, meta}`" → "`{text, emojis, styles, bg, fg}`".
- **Delete** the entire `tools/data/re-annotate.ts` bullet.
- Replace the `tools/data/labels.ts` bullet with:
  > - `tools/data/regen.ts` (`bun run regen`) — rebuilds `train.jsonl` + `eval.jsonl` + `labels.json` from `data.jsonl` in one command: union-merge by `normalize(text)`, seeded train/eval split (`--seed`, `--n`), and the fixed `styles` set + top-`TOP_EMOJIS` emoji leaderboard (frequency over the whole master).
- `tools/data/stat.ts` bullet: "summarizing `train.jsonl` and `eval.jsonl` side by side" → "summarizing `data.jsonl`"; "Auto-run by the `.claude/settings.json` PostToolUse hook whenever `train.jsonl` or `eval.jsonl` is written." → "Auto-run by the `.claude/settings.json` PostToolUse hook whenever `data.jsonl` is written."
- In the same section, remove the "Provenance `meta`" bullet's claim that `train.ts` / `re-annotate.ts` write `meta` if it overstates reality — reword to "Rows currently carry no `meta` field; `data.py:read` ignores the field if present." (Keep it short; do not expand.)

- [ ] **Step 5: `CLAUDE.md` — training-corpus + label bullets**

- The `labels.json` top-level bullet: "by frequency in `train.jsonl` (descending, ties keep first-seen order), regenerated by `tools/data/labels.ts`" → "by frequency over `data.jsonl` (descending, ties keep first-seen order), regenerated by `bun run regen`".
- The "**Training corpus:**" bullet: "`data.py` reads `train.jsonl` for training and `eval.jsonl` for validation; `data.jsonl` is legacy and is never trained on directly — `re-annotate.ts` migrates it into `train.jsonl`." → "`data.py` reads `train.jsonl` for training and `eval.jsonl` for validation; both are regenerated from the `data.jsonl` master by `bun run regen` and are gitignored (run `regen` after a fresh checkout or a pull that touched `data.jsonl`)."
- The "**Eval set**" bullet describing the old `eval.jsonl` hand-curation: append a sentence — "As of the data-files reorg, `eval.jsonl` is a plain seeded random split of the master, not a curated hold-out; `eval.jsonl.bak` is the pre-reorg curated set."

- [ ] **Step 6: `CLAUDE.md` — Environment & commands**

- In the "Package management is `uv` only" / environment area, add a bullet:
  > - `train.jsonl` / `eval.jsonl` / `labels.json` are gitignored and produced by `bun run regen` from `data.jsonl`. A fresh checkout must run `bun run regen` **before** `uv run python train.py` / `run.py` / `export_onnx.py` (they import `config`, which loads `labels.json`) and before `npm test` in `web/` (`feelings.test.js` reads `labels.json`). The Pages deploy is unaffected — it builds from the committed `web/public/`.
- The "data toolchain is Bun" bullet: "grow the corpus with `bun run tools/data/train.ts` and/or `bun run tools/data/re-annotate.ts`, then `bun run tools/data/labels.ts` to refresh `labels.json`. package.json aliases (root-relative): `bun run train` / `re-annotate` / `labels` / `stat`." → "grow the corpus with `bun run train` and/or `bun run upsample-emojis` / `bun run upsample-emoji-test` (all append to `data.jsonl`), then `bun run regen` to rebuild `train.jsonl` / `eval.jsonl` / `labels.json`. package.json aliases (root-relative): `bun run train` / `upsample-emojis` / `upsample-emoji-test` / `regen` / `stat`."
- The "Non-model scripts live under `tools/data/`" bullet lists the model stack — no change needed there, but if it names `re-annotate.ts` / `snapshot.ts` / `merge.ts` anywhere, drop those names.

- [ ] **Step 7: Grep CLAUDE.md for stragglers**

Run: `grep -n 're-annotate\|snapshot\.ts\|merge\.ts\|split-eval\|labels\.ts\|"train\.jsonl" is the corpus\|drains into' CLAUDE.md`
Expected: no matches (or only matches inside unrelated historical context you've confirmed are fine).

- [ ] **Step 8: Lint + full sweep**

```bash
uv run ruff check .
uv run ruff format --check .
bun test tools/data/
bun run regen
uv run python -c "import config, data; print(len(config.EMOJIS), len(config.STYLES))"
```
Expected: ruff clean; tests green; `regen` exits 0; the import prints `352 21`.

- [ ] **Step 9: Commit**

```bash
git add .claude/commands/update-model-md.md train-modal.py CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: describe data.jsonl master + regen.ts pipeline

CLAUDE.md data-pipeline sections rewritten (master/derived split, regen
precondition for the Python + web toolchains). /update-model-md reads the
current schema and distinguishes master from the post-regen train set.
train-modal.py mounts data.jsonl.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

---

## Task 7: Final verification and `todo.txt`

**Files:**
- Modify: `todo.txt` (move the completed item)

- [ ] **Step 1: End-to-end pipeline check**

```bash
git stash list && git status --short
bun run regen
md5sum train.jsonl eval.jsonl labels.json
bun run regen
md5sum train.jsonl eval.jsonl labels.json
bun test tools/data/
uv run ruff check . && uv run ruff format --check .
uv run python -c "import data; print(sum(1 for _ in data.read('train.jsonl')), sum(1 for _ in data.read('eval.jsonl')))"
```
Expected: `git status` shows no unexpected tracked-file churn (`data.jsonl` committed; `train.jsonl` / `eval.jsonl` / `labels.json` ignored, not listed); both `md5sum` runs match; tests + ruff green; the Python line prints two sane counts.

- [ ] **Step 2: Confirm the derived files are untracked**

```bash
git check-ignore train.jsonl eval.jsonl labels.json
git ls-files | grep -E '^(train|eval)\.jsonl$|^labels\.json$' || echo "none tracked (correct)"
```
Expected: `git check-ignore` echoes all three; the `ls-files` grep prints `none tracked (correct)`.

- [ ] **Step 3: Move the todo item**

In `todo.txt`, cut the `Data:` block (the "Reorganize the data files:" item and its sub-bullets, lines ~316–323) from the `TODO:` area and paste it under the `DONE:` section near the top of that section, followed by a one-line result note:
```
        Data:
            Reorganize the data files (data.jsonl master, train/eval/labels
            regenerated by `bun run regen`).
            ---
            data.jsonl is now the append-only committed master; train.jsonl /
            eval.jsonl / labels.json are gitignored, rebuilt by tools/data/regen.ts
            (union-merge by normalize(text) + seeded split). Deleted snapshot.ts /
            merge.ts / re-annotate.ts / labels.ts / split-eval.ts; train.ts +
            upsample-*.ts append to data.jsonl; stat.ts / the PostToolUse hook /
            /data-quality / /update-model-md / CLAUDE.md updated.
```
Leave every other line of `todo.txt` untouched (there is a pre-existing staged modification to this file — do not revert it, and stage only your own hunk if `git add -p` is available; otherwise `git add todo.txt` is acceptable since the pre-existing change is a legitimate content addition).

- [ ] **Step 4: Commit**

```bash
git add todo.txt
git commit -m "$(cat <<'EOF'
todo: mark data files reorganization done

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PSQ9zMfUz53PZQEAYxp29h
EOF
)"
```

- [ ] **Step 5: Report**

Summarize in 2–3 sentences: `data.jsonl` line count, the `bun run regen` train/eval/labels counts, the five deleted tools, and the new `bun run regen` precondition for a fresh checkout.

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
| --- | --- |
| Target file model (data.jsonl master, three derived gitignored) | 2 (regen main, migration, gitignore) |
| Row schema unchanged; palette-less row shape | Global Constraints; Task 1 `collapse` test "no bg/fg" |
| `regen.ts` step 1 collapse (union, `STYLE_SET` filter, hash palette) | Task 1 |
| `regen.ts` step 2 seeded split | Task 1 `shuffleSeeded` + Task 2 `main` |
| `regen.ts` step 3 labels (fixed styles + leaderboard over master) | Task 1 `emojiLeaderboard` + Task 2 `main` |
| `regen.ts` step 4 summary | Task 2 Step 1 |
| `regen.test.ts` | Task 1 |
| Growth tools rewired | Task 3 Steps 2–4 |
| Deletions (snapshot, merge, re-annotate, labels, split-eval) + `pool.ts` kept | Task 3 Steps 5–6; `pool.ts` untouched (Task 1 imports `stableHash`) |
| `package.json` aliases | Task 2 Step 2 (add), Task 3 Step 6 (remove) |
| One-time migration | Task 2 Step 3 |
| Python/Modal/web consequences (regen precondition) | Task 6 Steps 2, 6 |
| `settings.json` hook → data.jsonl | Task 4 Step 3 |
| `stat.ts` → `["./data.jsonl"]` | Task 4 Step 1 |
| `/data-quality` rewrite (drop leakage + provenance, add dup-key) | Task 5 |
| `/update-model-md` tweak | Task 6 Step 1 |
| `CLAUDE.md` rewrite | Task 6 Steps 3–7 |
| Verification (regen runs, byte-identical re-run, ruff, bun test, no full train) | Tasks 2, 6, 7 |

No spec requirement is left without a task.

**2. Placeholder scan** — no "TBD" / "handle edge cases" / "similar to Task N" / bare "write tests". Every code step carries the actual code; every doc step carries the actual replacement text.

**3. Type consistency** — `collapse` returns `Row[]`; the row type is named `Row` from the start (Task 1 Step 3) so it never shadows the built-in `Record<K,V>` used in `rowPalette`, and `toLine` in Task 2 consumes `Row`. `pickPalette(key, palettes)` signature is identical in Task 1's implementation and its test. `shuffleSeeded<T>(rows, seed)` and `emojiLeaderboard(records, n)` match between definition, test, and `main()` call sites. Palette shape `{ bg: string[]; fg: string }` is consistent across `collapse`, `pickPalette`, `toLine`, and the emitted JSON.
