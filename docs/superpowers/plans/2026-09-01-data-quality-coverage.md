# Data Quality: Emoji/Feeling Coverage + Provenance Metadata — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a topical axis to text generation, restore the rare-emoji / rare-feeling upsamplers on the current pipeline, and stamp every newly generated row with provenance metadata.

**Architecture:** Two new pure helper modules (`meta.ts`, `rarest.ts`) with `bun test` coverage. `train.ts` gains a `TOPICS` round-robin. Two new entrypoint scripts (`upsample-emojis.ts`, `upsample-feelings.ts`) select the rarest labels *from `labels.json`*, generate targeted texts, run the existing shared `annotate()` (open-set labels + `bg`/`fg` colors), and append to `train.jsonl`. `re-annotate.ts` and `config.ts` get small edits; `CLAUDE.md` and `package.json` are updated.

**Tech Stack:** Bun + TypeScript, Vercel AI SDK (`ai`), `p-queue`, `cli-progress`, `zod`. Tests via `bun test`. Python side (`data.py`) unchanged — only a compatibility check.

**Spec:** `docs/superpowers/specs/2026-09-01-data-quality-coverage-design.md`

## Global Constraints

- **No dedup, no pre-annotate length/normalize filter anywhere.** Every generated text goes to `annotate()`; every successfully annotated row is written (duplicate texts with different labels included). `data.py:read` is the only filter.
- **Only `labels.json` labels are targeted.** Upsamplers rank rarity over the `labels.json` emoji/feeling lists only.
- **Only newly written rows carry `meta`.** Never backfill existing `train.jsonl` rows.
- **No comments or docstrings in source.** Keep `type: ignore` / `noqa` / shebangs only. (Project convention.)
- **Explicit `git add <paths>` in every commit** — never `git add -A` / `git add .` (a background job also commits to this branch).
- All tool scripts assume repo root as CWD. Run every command from `/home/gilad/Work/emojic`.
- Package management: `uv` for Python, `bun` for `tools/`. Never `pip install`.
- `meta` row shape (exact):

```json
{"text":"...","feeling":"...","emoji":"...","bg":["#..","#.."],"fg":"#..","meta":{"src":"train","v":1,"at":"2026-09-01","model":"openai/gpt-5.6-luna","topic":"kitchen & cooking","params":{"batchSize":50,"minLen":4,"maxLen":48}}}
```

`topic` only on `train.ts` rows; `target_emoji` only on `upsample-emojis.ts` rows; `target_feeling` only on `upsample-feelings.ts` rows; undefined optionals omitted entirely.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tools/data/meta.ts` | create | `rowMeta(fields)` — build the `meta` object (adds `at`, `model`) |
| `tools/data/meta.test.ts` | create | `bun test` for `rowMeta` |
| `tools/data/rarest.ts` | create | `rarest(keys, counts, n)` — N lowest-count keys, ties keep `keys` order |
| `tools/data/rarest.test.ts` | create | `bun test` for `rarest` |
| `tools/data/train.ts` | modify | add `TOPICS` + `topicForBatch`, per-batch topic, `meta` on each row |
| `tools/data/upsample-emojis.ts` | create | rare **palette** emoji upsampler (exports `countPalette`) |
| `tools/data/upsample-emojis.test.ts` | create | `bun test` for `countPalette` |
| `tools/data/upsample-feelings.ts` | create | rare **palette** feeling upsampler (exports `countPaletteFeelings`) |
| `tools/data/upsample-feelings.test.ts` | create | `bun test` for `countPaletteFeelings` |
| `tools/data/re-annotate.ts` | modify | stamp `meta` on both emitted rows |
| `tools/data/config.ts` | modify | `TOP_EMOJIS` 150 → 200 |
| `package.json` | modify | add `upsample-emojis` / `upsample-feelings` aliases |
| `CLAUDE.md` | modify | document new scripts, topic axis, `meta` schema, `v` convention; fix stale label counts |

---

## Task 1: `rowMeta` metadata helper

**Files:**
- Create: `tools/data/meta.ts`
- Test: `tools/data/meta.test.ts`

**Interfaces:**
- Consumes: `MODEL` (string) from `tools/data/annotate.ts` (already exported).
- Produces:
  ```ts
  export function rowMeta(f: {
    src: string
    v: number
    topic?: string
    target_emoji?: string
    target_feeling?: string
    params: Record<string, unknown>
  }): Record<string, unknown>
  ```
  Result always has keys `src`, `v`, `at` (`YYYY-MM-DD`), `model`, `params`; adds `topic` / `target_emoji` / `target_feeling` only when that arg is not `undefined`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/meta.test.ts`:

```ts
import { test, expect } from "bun:test"

import { MODEL } from "./annotate.ts"
import { rowMeta } from "./meta.ts"

test("rowMeta fills at + model and passes params through", () => {
  const m = rowMeta({ src: "train", v: 1, params: { batchSize: 50 } })
  expect(m.src).toBe("train")
  expect(m.v).toBe(1)
  expect(m.model).toBe(MODEL)
  expect(m.params).toEqual({ batchSize: 50 })
  expect(m.at as string).toMatch(/^\d{4}-\d{2}-\d{2}$/)
})

test("rowMeta omits undefined optionals", () => {
  const m = rowMeta({ src: "train", v: 1, params: {} })
  expect("topic" in m).toBe(false)
  expect("target_emoji" in m).toBe(false)
  expect("target_feeling" in m).toBe(false)
})

test("rowMeta includes optionals when provided", () => {
  const m = rowMeta({
    src: "upsample-emojis",
    v: 1,
    target_emoji: "📺",
    topic: undefined,
    params: { voice: "a nurse" },
  })
  expect(m.target_emoji).toBe("📺")
  expect("topic" in m).toBe(false)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/meta.test.ts`
Expected: FAIL — `Cannot find module './meta.ts'`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/data/meta.ts`:

```ts
import { MODEL } from "./annotate.ts"

export function rowMeta(f: {
  src: string
  v: number
  topic?: string
  target_emoji?: string
  target_feeling?: string
  params: Record<string, unknown>
}): Record<string, unknown> {
  const m: Record<string, unknown> = {
    src: f.src,
    v: f.v,
    at: new Date().toISOString().slice(0, 10),
    model: MODEL,
    params: f.params,
  }
  if (f.topic !== undefined) m.topic = f.topic
  if (f.target_emoji !== undefined) m.target_emoji = f.target_emoji
  if (f.target_feeling !== undefined) m.target_feeling = f.target_feeling
  return m
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test tools/data/meta.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify `data.py` still loads a `meta`-tagged row**

Run:

```bash
uv run python -c "
import json, data
lb = json.load(open('labels.json'))
row = {'text':'watching tv tonight','feeling':lb['feelings'][0],'emoji':lb['emojis'][0],
       'bg':['#111111','#222222'],'fg':'#ffffff',
       'meta':{'src':'train','v':1,'at':'2026-09-01','model':'m','topic':'x','params':{}}}
open('/tmp/meta_check.jsonl','w').write(json.dumps(row)+'\n')
rows = list(data.read('/tmp/meta_check.jsonl'))
assert len(rows) == 1, rows
bad = dict(row); bad['emoji'] = 'ZZZ-not-an-emoji'
open('/tmp/meta_check.jsonl','w').write(json.dumps(bad)+'\n')
assert list(data.read('/tmp/meta_check.jsonl')) == [], 'out-of-palette row should be dropped'
print('data.py meta compatibility: OK')
"
```

Expected: prints `data.py meta compatibility: OK`.

- [ ] **Step 6: Commit**

```bash
git add tools/data/meta.ts tools/data/meta.test.ts
git commit -m "$(printf 'data: rowMeta provenance helper\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 2: `rarest` selection helper

**Files:**
- Create: `tools/data/rarest.ts`
- Test: `tools/data/rarest.test.ts`

**Interfaces:**
- Produces:
  ```ts
  export function rarest(keys: string[], counts: Map<string, number>, n: number): string[]
  ```
  Returns up to `n` entries of `keys` with the lowest `counts.get(key) ?? 0`, ascending; ties broken by original index in `keys`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/rarest.test.ts`:

```ts
import { test, expect } from "bun:test"

import { rarest } from "./rarest.ts"

test("rarest returns lowest-count keys ascending", () => {
  const counts = new Map([["a", 5], ["b", 1], ["c", 3]])
  expect(rarest(["a", "b", "c"], counts, 2)).toEqual(["b", "c"])
})

test("rarest breaks ties by key order", () => {
  const counts = new Map([["a", 2], ["b", 2], ["c", 2]])
  expect(rarest(["c", "a", "b"], counts, 2)).toEqual(["c", "a"])
})

test("rarest treats missing keys as 0 and caps at keys.length", () => {
  const counts = new Map([["a", 4]])
  expect(rarest(["a", "b", "c"], counts, 10)).toEqual(["b", "c", "a"])
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/rarest.test.ts`
Expected: FAIL — `Cannot find module './rarest.ts'`.

- [ ] **Step 3: Write minimal implementation**

Create `tools/data/rarest.ts`:

```ts
export function rarest(
  keys: string[],
  counts: Map<string, number>,
  n: number,
): string[] {
  return keys
    .map((k, i) => ({ k, i, c: counts.get(k) ?? 0 }))
    .sort((a, b) => a.c - b.c || a.i - b.i)
    .slice(0, n)
    .map((x) => x.k)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test tools/data/rarest.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/data/rarest.ts tools/data/rarest.test.ts
git commit -m "$(printf 'data: rarest(keys, counts, n) selection helper\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 3: `train.ts` topical axis + metadata

**Files:**
- Modify: `tools/data/train.ts`
- Test: `tools/data/train.test.ts` (create)

**Interfaces:**
- Consumes: `rowMeta` from `./meta.ts` (Task 1).
- Produces (new exports on `train.ts`):
  ```ts
  export const TOPICS: string[]
  export function topicForBatch(i: number): string   // TOPICS[i % TOPICS.length]
  ```

**Context — current `train.ts` shape (as committed):**
- `GEN_PROMPT` is a `const` string built from a line array (lines 17–26).
- `genBatch()` takes no args, returns `Promise<string[]>` (lines 28–39).
- Under `if (import.meta.main)`: `const texts: string[] = []`; `genQ.addAll(Array.from({ length: BATCH_COUNT }, () => async () => { texts.push(...(await genBatch())) ... }))`.
- Write loop (lines 79–92): `for (let i = 0; i < texts.length; i++)` → `JSON.stringify({ text: texts[i], feeling: label.feeling, emoji: label.emoji, bg: label.bg, fg: label.fg })`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/train.test.ts`:

```ts
import { test, expect } from "bun:test"

import { TOPICS, topicForBatch } from "./train.ts"

test("TOPICS is a non-trivial unique list", () => {
  expect(TOPICS.length).toBeGreaterThan(15)
  expect(new Set(TOPICS).size).toBe(TOPICS.length)
})

test("topicForBatch round-robins over TOPICS", () => {
  expect(topicForBatch(0)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length + 1)).toBe(TOPICS[1])
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/train.test.ts`
Expected: FAIL — `TOPICS`/`topicForBatch` not exported.

- [ ] **Step 3: Add `TOPICS` + `topicForBatch`**

In `tools/data/train.ts`, directly after the `MAX_LEN` const, add:

```ts
export const TOPICS = [
  "food & cooking",
  "chores & housework",
  "commute & getting around",
  "weather & seasons",
  "pets & animals",
  "phones, apps & devices",
  "shopping & money",
  "tv, film & streaming",
  "video games",
  "sport & exercise",
  "health, sleep & the body",
  "school, class & studying",
  "work & the office",
  "music & gigs",
  "clothes & how things look",
  "events, parties & celebrations",
  "hobbies, making & fixing things",
  "outdoors, parks & nature",
  "plans, scheduling & logistics",
  "eating out & food delivery",
  "cars, bikes & public transport",
  "news & things happening",
  "family, friends & relationships",
  "random small talk",
]

export function topicForBatch(i: number): string {
  return TOPICS[i % TOPICS.length]
}
```

- [ ] **Step 4: Make `genBatch` topic-aware**

Replace the `GEN_PROMPT` const and `genBatch` function with:

```ts
function genPrompt(topic: string): string {
  return [
    `Write ${BATCH_SIZE} short text messages, one per line.`,
    `Each message between ${MIN_LEN} and ${MAX_LEN} characters.`,
    `Every message is about ${topic}. Do not announce the topic or use it as a`,
    "label; let it come through naturally in what is said.",
    "Across the whole set, still cover a wide range: many different senders and",
    "personalities, every mood (positive, negative, flat), every intent",
    "(statements, questions, requests, reactions, reminders, small talk), and",
    "every length from very short to near the maximum.",
    "Do not lean on any single persona, tone, or sentence shape.",
    "No numbering, no bullets, no quotes, no emoji, no commentary.",
  ].join("\n")
}

async function genBatch(topic: string): Promise<string[]> {
  const { text } = await generateText({ model: MODEL, prompt: genPrompt(topic) })
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}
```

- [ ] **Step 5: Carry `topic` through the run and onto each row**

In the `if (import.meta.main)` block:

1. Change the accumulator:
   ```ts
   const texts: { text: string; topic: string }[] = []
   ```
2. Change the generation `addAll` to pass the batch index and topic:
   ```ts
   genQ.addAll(
     Array.from({ length: BATCH_COUNT }, (_, i) => async () => {
       const topic = topicForBatch(i)
       try {
         for (const t of await genBatch(topic)) texts.push({ text: t, topic })
       } catch (err) {
         console.warn(`\n  gen batch failed: ${err}`)
       }
       genBar.increment()
     }),
   )
   ```
3. Change the annotate call to pass plain strings:
   ```ts
   const labels = await annotate(
     texts.map((t) => t.text),
     () => annBar.increment(),
   )
   ```
4. Change the write loop body to read `.text` / `.topic` and add `meta` (add the import `import { rowMeta } from "./meta.ts"` at the top with the other local imports):
   ```ts
   for (let i = 0; i < texts.length; i++) {
     const label = labels.get(i)
     if (!label) continue
     lines.push(
       JSON.stringify({
         text: texts[i].text,
         feeling: label.feeling,
         emoji: label.emoji,
         bg: label.bg,
         fg: label.fg,
         meta: rowMeta({
           src: "train",
           v: 1,
           topic: texts[i].topic,
           params: { batchSize: BATCH_SIZE, minLen: MIN_LEN, maxLen: MAX_LEN },
         }),
       }),
     )
   }
   ```

Leave `annBar.start(annotateBatchCount(texts.length), 0)` as-is — `texts.length` is still the text count.

- [ ] **Step 6: Run the test to verify it passes**

Run: `bun test tools/data/train.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 7: Verify the entrypoint still bundles**

Run: `bun build tools/data/train.ts --target bun --outdir /tmp/emojic-build`
Expected: `Bundled ... [Nms]` with no error. (Does not execute the script.)

- [ ] **Step 8: Commit**

```bash
git add tools/data/train.ts tools/data/train.test.ts
git commit -m "$(printf 'data: topical axis + provenance meta in train.ts\n\nEach generation batch is anchored to one of 24 concrete topics\n(round-robin) so object/noun emojis get corpus coverage. Every row\nnow carries meta{src,v,at,model,topic,params}.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 4: `upsample-emojis.ts` — rare palette-emoji upsampler

**Files:**
- Create: `tools/data/upsample-emojis.ts`
- Test: `tools/data/upsample-emojis.test.ts`

**Interfaces:**
- Consumes: `annotate`, `annotateBatchCount`, `MODEL` from `./annotate.ts`; `appendJsonl`, `readJsonl` from `./io.ts`; `rarest` from `./rarest.ts` (Task 2); `rowMeta` from `./meta.ts` (Task 1).
- Produces:
  ```ts
  export function countPalette(
    rows: { emoji: string }[],
    palette: string[],
  ): Map<string, number>
  ```
  Map has one entry per `palette` emoji (0 for absent), counting only rows whose `emoji` is in `palette`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/upsample-emojis.test.ts`:

```ts
import { test, expect } from "bun:test"

import { countPalette } from "./upsample-emojis.ts"

test("countPalette counts only palette emojis, zero-fills the rest", () => {
  const rows = [
    { emoji: "📺" },
    { emoji: "📺" },
    { emoji: "🍕" },
    { emoji: "🛰️" },
  ]
  const counts = countPalette(rows, ["📺", "🍕", "🚗"])
  expect(counts.get("📺")).toBe(2)
  expect(counts.get("🍕")).toBe(1)
  expect(counts.get("🚗")).toBe(0)
  expect(counts.has("🛰️")).toBe(false)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/upsample-emojis.test.ts`
Expected: FAIL — `Cannot find module './upsample-emojis.ts'`.

- [ ] **Step 3: Write the script**

Create `tools/data/upsample-emojis.ts`:

```ts
import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { rowMeta } from "./meta.ts"
import { rarest } from "./rarest.ts"

const TRAIN = "./train.jsonl"
const LABELS = "./labels.json"

const RARE_EMOJI_COUNT = 60
const TEXTS_PER_EMOJI = 40
const MAX_RAW_LEN = 50
const GEN_CONCURRENCY = 20

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

export function countPalette(
  rows: { emoji: string }[],
  palette: string[],
): Map<string, number> {
  const counts = new Map<string, number>(palette.map((e) => [e, 0]))
  for (const r of rows) {
    const c = counts.get(r.emoji)
    if (c !== undefined) counts.set(r.emoji, c + 1)
  }
  return counts
}

async function genBatch(voice: string, emoji: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${TEXTS_PER_EMOJI} short WhatsApp-style messages as if sent by ${voice}.`,
      `Every message must read naturally as one that would be sent together with the emoji ${emoji} -- its subject, mood, or occasion should fit that emoji.`,
      `Do not put any emoji in the output, and never name or describe the emoji.`,
      `One message per line. No numbering, no bullets, no quotes, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary tone and intent: quick updates, dry humor, complaints, questions, sudden news, invitations, low-effort replies.`,
      `Sound real and specific.`,
    ].join("\n"),
  })
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

if (import.meta.main) {
  const labels = JSON.parse(await Bun.file(LABELS).text()) as {
    feelings: string[]
    emojis: string[]
  }
  const palette = labels.emojis

  const rows = await readJsonl<{ emoji: string }>(TRAIN)
  const counts = countPalette(rows, palette)
  const targets = rarest(palette, counts, RARE_EMOJI_COUNT)
  console.log(
    `${palette.length} palette emojis -> upsampling ${targets.length} rarest ` +
      `(${counts.get(targets[0])}..${counts.get(targets[targets.length - 1])} rows each)`,
  )

  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)

  const cands: { text: string; target: string; voice: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map((emoji) => async () => {
      const voice = pickVoice()
      try {
        for (const t of await genBatch(voice, emoji)) {
          cands.push({ text: t, target: emoji, voice })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${emoji}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()

  console.log(`\n${cands.length} texts generated, annotating`)

  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(cands.length), 0)
  const annotated = await annotate(
    cands.map((c) => c.text),
    () => annBar.increment(),
  )
  annBar.stop()

  const lines: string[] = []
  for (let i = 0; i < cands.length; i++) {
    const label = annotated.get(i)
    if (!label) continue
    lines.push(
      JSON.stringify({
        text: cands[i].text,
        feeling: label.feeling,
        emoji: label.emoji,
        bg: label.bg,
        fg: label.fg,
        meta: rowMeta({
          src: "upsample-emojis",
          v: 1,
          target_emoji: cands[i].target,
          params: { voice: cands[i].voice, textsPerEmoji: TEXTS_PER_EMOJI },
        }),
      }),
    )
  }
  await appendJsonl(TRAIN, lines)

  console.log("\n--- summary ---")
  console.log(`targets (rarest)     : ${targets.length}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`annotated -> train   : ${lines.length}`)
  console.log(`dropped (no label)   : ${cands.length - lines.length}`)
  process.exit(0)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test tools/data/upsample-emojis.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Verify the entrypoint bundles**

Run: `bun build tools/data/upsample-emojis.ts --target bun --outdir /tmp/emojic-build`
Expected: bundled, no error.

- [ ] **Step 6: Commit**

```bash
git add tools/data/upsample-emojis.ts tools/data/upsample-emojis.test.ts
git commit -m "$(printf 'data: upsample-emojis.ts (rare palette-emoji upsampler)\n\nPorts the deleted emoji2feeling.ts onto the current pipeline: ranks\nrarity over labels.json emojis, generates targeted texts, runs the\nshared open-set annotate() for labels + colors, appends to\ntrain.jsonl with meta{src,target_emoji}. No dedup, no pre-filter.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 5: `upsample-feelings.ts` — rare palette-feeling upsampler

**Files:**
- Create: `tools/data/upsample-feelings.ts`
- Test: `tools/data/upsample-feelings.test.ts`

**Interfaces:**
- Consumes: same imports as Task 4.
- Produces:
  ```ts
  export function countPaletteFeelings(
    rows: { emoji: string; feeling: string }[],
    feelings: string[],
    emojiPalette: Set<string>,
  ): Map<string, number>
  ```
  Map has one entry per `feelings` entry (0 for absent), counting only rows whose `feeling` is in `feelings` **and** whose `emoji` is in `emojiPalette`.

- [ ] **Step 1: Write the failing test**

Create `tools/data/upsample-feelings.test.ts`:

```ts
import { test, expect } from "bun:test"

import { countPaletteFeelings } from "./upsample-feelings.ts"

test("countPaletteFeelings counts rows in-palette on both axes", () => {
  const rows = [
    { emoji: "📺", feeling: "Calm" },
    { emoji: "📺", feeling: "Calm" },
    { emoji: "🛰️", feeling: "Calm" },
    { emoji: "📺", feeling: "Rapturous" },
  ]
  const counts = countPaletteFeelings(rows, ["Calm", "Sad"], new Set(["📺"]))
  expect(counts.get("Calm")).toBe(2)
  expect(counts.get("Sad")).toBe(0)
  expect(counts.has("Rapturous")).toBe(false)
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `bun test tools/data/upsample-feelings.test.ts`
Expected: FAIL — `Cannot find module './upsample-feelings.ts'`.

- [ ] **Step 3: Write the script**

Create `tools/data/upsample-feelings.ts`:

```ts
import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"

import { MODEL, annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, readJsonl } from "./io.ts"
import { rowMeta } from "./meta.ts"
import { rarest } from "./rarest.ts"

const TRAIN = "./train.jsonl"
const LABELS = "./labels.json"

const FEELINGS_PER_RUN = 3
const BATCHES_PER_FEELING = 20
const GEN_BATCH_SIZE = 100
const MAX_RAW_LEN = 50
const GEN_CONCURRENCY = 20

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

export function countPaletteFeelings(
  rows: { emoji: string; feeling: string }[],
  feelings: string[],
  emojiPalette: Set<string>,
): Map<string, number> {
  const counts = new Map<string, number>(feelings.map((f) => [f, 0]))
  for (const r of rows) {
    if (counts.has(r.feeling) && emojiPalette.has(r.emoji)) {
      counts.set(r.feeling, counts.get(r.feeling)! + 1)
    }
  }
  return counts
}

async function genBatch(voice: string, feeling: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${GEN_BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
      `Every message must unmistakably convey the feeling "${feeling}" -- someone reading it cold, with no context, should name that feeling. Convey it through what is said and how, never by naming the feeling.`,
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary the wording, situation and sentence shape, but keep every line firmly in the target feeling.`,
      `Sound real and specific.`,
    ].join("\n"),
  })
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "")
        .replace(/^["'`]+|["'`]+$/g, "")
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

if (import.meta.main) {
  const labels = JSON.parse(await Bun.file(LABELS).text()) as {
    feelings: string[]
    emojis: string[]
  }
  const feelings = labels.feelings
  const emojiPalette = new Set(labels.emojis)

  const rows = await readJsonl<{ emoji: string; feeling: string }>(TRAIN)
  const counts = countPaletteFeelings(rows, feelings, emojiPalette)
  const targets = rarest(feelings, counts, FEELINGS_PER_RUN)
  console.log(
    `feeling coverage: ${feelings.map((f) => `${f}=${counts.get(f)}`).join(" ")}`,
  )
  console.log(`upsampling rarest: ${targets.join(", ")}`)

  const batchCount = targets.length * BATCHES_PER_FEELING
  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(batchCount, 0)

  const cands: { text: string; target: string; voice: string }[] = []
  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    Array.from({ length: batchCount }, (_, i) => async () => {
      const feeling = targets[i % targets.length]
      const voice = pickVoice()
      try {
        for (const t of await genBatch(voice, feeling)) {
          cands.push({ text: t, target: feeling, voice })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${feeling}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()

  console.log(`\n${cands.length} texts generated, annotating`)

  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(cands.length), 0)
  const annotated = await annotate(
    cands.map((c) => c.text),
    () => annBar.increment(),
  )
  annBar.stop()

  const lines: string[] = []
  for (let i = 0; i < cands.length; i++) {
    const label = annotated.get(i)
    if (!label) continue
    lines.push(
      JSON.stringify({
        text: cands[i].text,
        feeling: label.feeling,
        emoji: label.emoji,
        bg: label.bg,
        fg: label.fg,
        meta: rowMeta({
          src: "upsample-feelings",
          v: 1,
          target_feeling: cands[i].target,
          params: {
            voice: cands[i].voice,
            batchesPerFeeling: BATCHES_PER_FEELING,
          },
        }),
      }),
    )
  }
  await appendJsonl(TRAIN, lines)

  console.log("\n--- summary ---")
  console.log(`targets (rarest)     : ${targets.join(", ")}`)
  console.log(`generated            : ${cands.length}`)
  console.log(`annotated -> train   : ${lines.length}`)
  console.log(`dropped (no label)   : ${cands.length - lines.length}`)
  process.exit(0)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `bun test tools/data/upsample-feelings.test.ts`
Expected: PASS (1 test).

- [ ] **Step 5: Verify the entrypoint bundles**

Run: `bun build tools/data/upsample-feelings.ts --target bun --outdir /tmp/emojic-build`
Expected: bundled, no error.

- [ ] **Step 6: Commit**

```bash
git add tools/data/upsample-feelings.ts tools/data/upsample-feelings.test.ts
git commit -m "$(printf 'data: upsample-feelings.ts (rare palette-feeling upsampler)\n\nPorts the deleted feeling2emoji.ts onto the current pipeline: ranks\nrarity over labels.json feelings (rows in-palette on both axes),\ngenerates feeling-targeted texts, runs the shared open-set annotate()\nfor labels + colors, appends to train.jsonl with meta{src,target_feeling}.\nDrops the old closed-list annotation and drift-reject step.\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 6: `re-annotate.ts` — provenance metadata

**Files:**
- Modify: `tools/data/re-annotate.ts`

**Interfaces:**
- Consumes: `rowMeta` from `./meta.ts` (Task 1).

**Context — current `re-annotate.ts` shape (as committed):**
- Lines 84–95: `const emit = (feeling: string, emoji: string) => appended.push(JSON.stringify({ text: texts[p], feeling, emoji, bg: label.bg, fg: label.fg }))`.
- `emit` is called once with the new labels (line 95) and, when labels changed, again with the old labels (line 102).
- `SAMPLE_SIZE` is a module const (line 12).

- [ ] **Step 1: Add the import**

At the top of `tools/data/re-annotate.ts`, with the other local imports:

```ts
import { rowMeta } from "./meta.ts"
```

- [ ] **Step 2: Stamp `meta` in `emit`**

Replace the `emit` arrow (lines ~85–94) with:

```ts
    const emit = (feeling: string, emoji: string) =>
      appended.push(
        JSON.stringify({
          text: texts[p],
          feeling,
          emoji,
          bg: label.bg,
          fg: label.fg,
          meta: rowMeta({
            src: "re-annotate",
            v: 1,
            params: { sampleSize: SAMPLE_SIZE },
          }),
        }),
      )
```

Both the new-label and old-label rows go through `emit`, so both get `meta`.

- [ ] **Step 3: Verify the entrypoint bundles**

Run: `bun build tools/data/re-annotate.ts --target bun --outdir /tmp/emojic-build`
Expected: bundled, no error.

- [ ] **Step 4: Commit**

```bash
git add tools/data/re-annotate.ts
git commit -m "$(printf 'data: stamp provenance meta in re-annotate.ts\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Task 7: Config, package.json aliases, CLAUDE.md

**Files:**
- Modify: `tools/data/config.ts`
- Modify: `package.json`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Bump the emoji leaderboard size**

In `tools/data/config.ts`, change:

```ts
export const TOP_EMOJIS = 150
```

to:

```ts
export const TOP_EMOJIS = 200
```

Leave `TOP_FEELINGS = 32` unchanged.

- [ ] **Step 2: Add package.json aliases**

In `package.json`, in `"scripts"`, after the `"re-annotate"` line, add:

```json
    "upsample-emojis": "bun run tools/data/upsample-emojis.ts",
    "upsample-feelings": "bun run tools/data/upsample-feelings.ts",
```

Verify the file still parses: `node -e "JSON.parse(require('fs').readFileSync('package.json','utf8')); console.log('package.json ok')"`

- [ ] **Step 3: Verify config imports cleanly**

Run: `bun -e "import('./tools/data/config.ts').then(c => console.log('TOP_EMOJIS', c.TOP_EMOJIS, 'TOP_FEELINGS', c.TOP_FEELINGS))"`
Expected: `TOP_EMOJIS 200 TOP_FEELINGS 32`.

- [ ] **Step 4: Update CLAUDE.md**

In `CLAUDE.md`, make these edits to the data-pipeline bullets:

1. In the `labels.json` description, change "`feelings` = top 10, `emojis` = top 100" to "`feelings` = top 32, `emojis` = top 200" (values from `tools/data/config.ts`).

2. In the `tools/data/train.ts` bullet, add a sentence: "Phase 1 batches are each anchored round-robin to one of `TOPICS` (24 concrete domains — food, chores, tv & film, pets, tech, …) so object/noun emojis get corpus coverage; the batch's topic is recorded on every row it produces."

3. Add two new bullets under the data-pipeline section, after `re-annotate.ts`:

   - `tools/data/upsample-emojis.ts` (`bun run upsample-emojis`) — reads `train.jsonl`, counts each `labels.json` emoji's rows (palette emojis absent count 0; non-palette emojis ignored), takes the `RARE_EMOJI_COUNT` (60) rarest, generates `TEXTS_PER_EMOJI` (40) voice-spread texts per emoji (emoji never named), runs the shared `annotate()` (open-set labels + `bg`/`fg`), and appends every annotated row to `train.jsonl`. No dedup, no length pre-filter — `data.py` filters at load.
   - `tools/data/upsample-feelings.ts` (`bun run upsample-feelings`) — same shape, seeded by the `FEELINGS_PER_RUN` (3) rarest `labels.json` feelings (counting only rows whose emoji is also in the palette), `BATCHES_PER_FEELING` (20) batches of `GEN_BATCH_SIZE` (100) per feeling.

4. Add a bullet describing the `meta` field: "Every row written by `train.ts` / `upsample-emojis.ts` / `upsample-feelings.ts` / `re-annotate.ts` carries `meta`: `{src, v, at (YYYY-MM-DD), model, params}` plus `topic` (train only) or `target_emoji` / `target_feeling` (upsamplers only). `v` is the per-script logic version — bump it by hand when a script's generation/annotation logic changes. Pre-existing rows have no `meta`; `data.py:read` ignores the field either way."

5. In the "data toolchain is Bun" commands paragraph, add `upsample-emojis` / `upsample-feelings` to the list of `bun run` aliases.

- [ ] **Step 5: Full test sweep**

Run: `bun test tools/data/`
Expected: all tests from Tasks 1–5 PASS, 0 fail.

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no changes / no errors (no Python was edited; this is a regression guard).

- [ ] **Step 6: Commit**

```bash
git add tools/data/config.ts package.json CLAUDE.md
git commit -m "$(printf 'data: TOP_EMOJIS=200, upsample-* aliases, CLAUDE.md\n\nCo-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01WwKfus26Rt3GrhEQNT2e2K')"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §1 `meta.ts` / `rowMeta` / row shape / `data.py` compat | Task 1 |
| §2 `train.ts` topical axis | Task 3 |
| §3 `upsample-emojis.ts` | Task 4 |
| §4 `upsample-feelings.ts` | Task 5 |
| §5 `re-annotate.ts` metadata | Task 6 |
| §6 `config.ts` `TOP_EMOJIS` 200 | Task 7 |
| §7 `package.json` aliases | Task 7 |
| §8 `CLAUDE.md` | Task 7 |
| `rarest` helper (spec data-flow, §3/§4) | Task 2 |
| Verification (`bun test`, `ruff`, `data.py` parse) | Task 1 step 5, Task 7 step 5 |

No gaps.

**2. Placeholder scan**

No "TBD"/"handle edge cases"/"similar to Task N" — every code step carries full source. `bun build --outdir /tmp/emojic-build` is a real command. `/tmp/emojic-build` is scratch and may be left in place.

**3. Type consistency**

- `rowMeta(f: { src; v; topic?; target_emoji?; target_feeling?; params })` — identical call shape in Tasks 3, 4, 5, 6.
- `rarest(keys: string[], counts: Map<string, number>, n: number): string[]` — defined Task 2, called identically in Tasks 4 and 5.
- `countPalette(rows, palette)` (Task 4) and `countPaletteFeelings(rows, feelings, emojiPalette)` (Task 5) — distinct names, each tested against its own signature.
- `annotate(texts: string[], onBatchDone?)` and `annotateBatchCount(n)` — used as already exported from `annotate.ts` (verified against the committed file).
- `readJsonl<T>(path)`, `appendJsonl(path, rows)` — used as exported from `io.ts` (verified).
- `MODEL` — string, imported from `annotate.ts` in `meta.ts`, Task 4, Task 5 (verified exported).
- `labels.json` shape `{ feelings: string[]; emojis: string[] }` — read the same way in Tasks 4 and 5.

Consistent.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-01-data-quality-coverage.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
