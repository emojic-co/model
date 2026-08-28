/**
 * Synthetic dataset pipeline for `emojic`.
 *
 * One run performs three steps:
 *
 *   1. Generate raw WhatsApp-style texts (no labels, no schema) across a wide
 *      spread of topics and speaker voices. Lines are deduplicated against each
 *      other and against the existing data.jsonl using the same normalized key
 *      as main.py. Survivors are written to raw.txt.
 *   2. Annotate every raw text (batches of 10, like fix.ts): one feeling from
 *      the closed labels.json set and the single best-fit emoji (free choice).
 *      Merged {emoji, feeling, text} records are streamed to data.jsonl.tmp.
 *   3. Settle the label set. A record is kept if its emoji is already in
 *      labels.json, or if a *new* emoji reaches >= 1% of this run's records
 *      (~10). New emojis that clear the bar are appended to labels.json;
 *      feelings are a fixed closed set and never grow. labels.json is written
 *      *before* the new rows are appended to data.jsonl, so a crash can never
 *      leave data.jsonl referencing an emoji that labels.json lacks.
 *
 * On success raw.txt and data.jsonl.tmp are deleted: data.jsonl + labels.json
 * are the only ground truth and the input to the next run. A crashed run
 * resumes -- an existing raw.txt skips step 1, and rows already in
 * data.jsonl.tmp are not re-annotated.
 *
 * Run:
 *   bun run gen_data.ts
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { existsSync } from "node:fs"
import { appendFile, readFile, rm, writeFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const MAX_TEXT_LEN = 42
const DATA = "./data.jsonl"
const LABELS = "./labels.json"
const RAW = "./raw.txt"
const TMP = "./data.jsonl.tmp"

// Longest text we keep. main.py's encode() currently truncates at
// config.MAX_TEXT_LEN (48); we store up to 64 so the corpus survives a future
// MAX_TEXT_LEN bump without regeneration.

const TARGET_TEXTS = 1000
const GEN_BATCH = 25
const GEN_CONCURRENCY = 10
const ANNOTATE_BATCH = 10
const ANNOTATE_CONCURRENCY = 10
const NEW_EMOJI_MIN_FREQ = 0.01

// --- normalized dedup key: mirror of main.py's normalize() ------------------
const VOCAB = new Set("abcdefghijklmnopqrstuvwxyz!?:()@$%&* ")
function normalize(text: string): string {
  const t = text.replace(/\s+/g, " ").trim().toLowerCase()
  return [...t].filter((c) => VOCAB.has(c)).join("")
}

type Row = { emoji: string; feeling: string; text: string }

async function readJsonl(path: string): Promise<Row[]> {
  const raw = await readFile(path, "utf8")
  return raw
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Row)
}

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

async function pMap<T>(
  items: T[],
  fn: (x: T, i: number) => Promise<void>,
  concurrency: number,
): Promise<void> {
  let idx = 0
  async function worker() {
    while (idx < items.length) {
      const cur = idx++
      await fn(items[cur], cur)
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, items.length) }, worker),
  )
}

function progressBar(
  label: string,
  total: number,
  unit = "batches",
): cliProgress.SingleBar {
  const bar = new cliProgress.SingleBar(
    {
      format: `${label} |{bar}| {percentage}% | {value}/{total} ${unit} | ETA: {eta}s`,
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(total, 0)
  return bar
}

// ---------------------------------------------------------------- step 1: gen

// Topic + speaker-voice pools. Each generation batch draws one of each so the
// corpus spans ages, interests, and registers rather than one default persona.
const TOPICS = [
  "sports", "a football match", "basketball", "going for a run", "the gym",
  "music", "a concert", "a new album", "movies", "a TV series",
  "gaming", "politics", "the news", "the weather", "work",
  "a job interview", "school", "exams", "family", "the kids",
  "a partner", "dating", "friends", "a night out", "cooking",
  "a food delivery", "a restaurant", "travel plans", "a flight delay",
  "a road trip", "pets", "the dog", "money", "rent", "shopping",
  "a package that's late", "health", "a doctor visit", "the commute",
  "traffic", "a birthday", "a wedding", "moving house", "a DIY project",
  "gardening", "a book", "art", "photography", "the beach", "a hangover",
]

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
]

async function genBatch(topic: string, voice: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${GEN_BATCH} short WhatsApp-style messages as if sent by ${voice}, loosely about ${topic}.`,
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_TEXT_LEN} characters.`,
      `Vary tone and intent: quick updates, dry humor, complaints, questions, sudden news, invitations, low-effort replies.`,
      `Make roughly a quarter of the messages express a feeling by negating one: "not happy about this", "wasn't excited tbh", "no longer angry", "cant say im sad", "not that calm rn". Negate different feelings, not just one.`,
      `Sound real and specific. Avoid clichés and near-duplicates.`,
    ].join("\n"),
  })
  return text
    .split("\n")
    .map((l) =>
      l
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "") // strip a list marker
        .replace(/^["'`]+|["'`]+$/g, "") // strip wrapping quotes/backticks
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

async function step1(): Promise<string[]> {
  if (existsSync(RAW)) {
    const kept = (await readFile(RAW, "utf8"))
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean)
    console.log(`raw.txt present -> reusing ${kept.length} texts (step 1 skipped)`)
    return kept
  }

  const existing = new Set<string>()
  if (existsSync(DATA)) {
    for (const r of await readJsonl(DATA)) existing.add(normalize(r.text))
  }

  const nBatches = Math.ceil(TARGET_TEXTS / GEN_BATCH)
  const jobs = Array.from({ length: nBatches }, () => ({
    topic: TOPICS[Math.floor(Math.random() * TOPICS.length)],
    voice: VOICES[Math.floor(Math.random() * VOICES.length)],
  }))

  const seen = new Set<string>()
  const kept: string[] = []
  const bar = progressBar("generating", nBatches)
  await pMap(
    jobs,
    async (job) => {
      try {
        for (const line of await genBatch(job.topic, job.voice)) {
          const n = normalize(line)
          if (!n || n.length > MAX_RAW_LEN) continue
          if (existing.has(n) || seen.has(n)) continue
          seen.add(n)
          kept.push(line)
        }
      } catch (err) {
        console.warn(`\n  gen batch (${job.topic}) failed: ${err}`)
      }
      bar.increment()
    },
    GEN_CONCURRENCY,
  )
  bar.stop()

  await writeFile(RAW, kept.join("\n") + "\n")
  console.log(`step 1: ${kept.length} unique new texts -> ${RAW}`)
  return kept
}

// ------------------------------------------------------------ step 2: annotate

const Annotation = z.object({
  id: z.number(),
  feeling: z.string(),
  emoji: z.string(),
})

/**
 * Annotate one batch. Returns id -> {feeling, emoji} for every id the model
 * answered. Retries once on a shape/length mismatch or an API error; whatever
 * is still missing after that is logged and left out.
 */
async function annotateBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, { feeling: string; emoji: string }>> {
  const ids = new Set(batch.map((b) => b.id))
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [
          "You are an annotator. For each message below, decide:",
          `1. feeling: choose exactly one from this list: ${feelings.join(", ")}`,
          "2. emoji: the single emoji that best fits the message. Any emoji is allowed - pick the most fitting one, not a safe default.",
          "",
          "Return exactly one annotation object per input message, echoing its id.",
          "Do not add, drop, reorder, or merge items.",
          'Format: {"annotations": [{"id": 0, "feeling": "Happy", "emoji": "\u{1F600}"}]}',
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })

      const parsed = z
        .object({ annotations: z.array(Annotation) })
        .parse(output)

      const byId = new Map<number, { feeling: string; emoji: string }>()
      for (const a of parsed.annotations) {
        if (ids.has(a.id)) {
          byId.set(a.id, { feeling: a.feeling.trim(), emoji: a.emoji.trim() })
        }
      }

      const missing = batch.filter((b) => !byId.has(b.id))
      if (missing.length === 0) return byId
      if (attempt === 1) {
        for (const m of missing) {
          console.warn(`\n  dropped id ${m.id}: no annotation returned`)
        }
        return byId
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`\n  batch of ${batch.length} failed twice, skipped: ${err}`)
        return new Map()
      }
    }
  }
  return new Map()
}

async function step2(texts: string[], feelings: string[]): Promise<Row[]> {
  const done = new Set<string>()
  if (existsSync(TMP)) {
    for (const r of await readJsonl(TMP)) done.add(normalize(r.text))
    console.log(`step 2: ${done.size} rows already in ${TMP}, resuming`)
  }

  const todo = texts
    .map((text, id) => ({ id, text }))
    .filter((t) => !done.has(normalize(t.text)))
  const batches = chunk(todo, ANNOTATE_BATCH)

  // Serialize appends so concurrent workers don't interleave partial lines.
  let writeChain: Promise<unknown> = Promise.resolve()
  const append = (s: string) => {
    writeChain = writeChain.then(() => appendFile(TMP, s))
    return writeChain
  }

  const bar = progressBar("annotating", batches.length)
  await pMap(
    batches,
    async (batch) => {
      const byId = await annotateBatch(batch, feelings)
      const lines = batch
        .map((b) => {
          const a = byId.get(b.id)
          return a
            ? JSON.stringify({ emoji: a.emoji, feeling: a.feeling, text: b.text })
            : null
        })
        .filter(Boolean)
        .join("\n")
      if (lines) await append(lines + "\n")
      bar.increment()
    },
    ANNOTATE_CONCURRENCY,
  )
  await writeChain
  bar.stop()

  return readJsonl(TMP)
}

// -------------------------------------------------- step 3: labels + keep + write

async function step3(records: Row[]): Promise<void> {
  const bar = progressBar("settling ", 5, "phases")

  // 1. current label set
  const { feelings, emojis } = z
    .object({ feelings: z.array(z.string()), emojis: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8")))
  const feelingSet = new Set(feelings)
  const emojiSet = new Set(emojis)
  bar.increment()

  // 2. tally this run's emoji frequencies
  const counts = new Map<string, number>()
  for (const r of records) counts.set(r.emoji, (counts.get(r.emoji) ?? 0) + 1)
  const threshold = records.length * NEW_EMOJI_MIN_FREQ
  bar.increment()

  // 3. resolve the keep set and filter
  const newEmojis = [...counts.entries()]
    .filter(([e, n]) => !emojiSet.has(e) && n >= threshold)
    .sort((a, b) => b[1] - a[1])
    .map(([e]) => e)
  const keepEmoji = new Set([...emojiSet, ...newEmojis])
  const kept = records.filter(
    (r) => keepEmoji.has(r.emoji) && feelingSet.has(r.feeling),
  )
  bar.increment()

  // 4. labels.json FIRST: an emoji in labels.json with no rows yet is harmless;
  // a data.jsonl row whose emoji is absent from labels.json breaks main.py. A
  // crash between these two writes must only ever leave the harmless state.
  if (newEmojis.length) {
    await writeFile(
      LABELS,
      JSON.stringify({ feelings, emojis: [...emojis, ...newEmojis] }, null, 2) +
      "\n",
    )
  }
  bar.increment()

  // 5. append the kept rows
  if (kept.length) {
    await appendFile(DATA, kept.map((r) => JSON.stringify(r)).join("\n") + "\n")
  }
  bar.increment()
  bar.stop()

  const droppedEmoji = [...counts.entries()]
    .filter(([e]) => !keepEmoji.has(e))
    .sort((a, b) => b[1] - a[1])
  console.log("\n--- summary ---")
  console.log(`annotated records   : ${records.length}`)
  console.log(`kept -> data.jsonl  : ${kept.length}`)
  console.log(
    `dropped             : ${records.length - kept.length} ` +
    `(new emoji <${NEW_EMOJI_MIN_FREQ * 100}% and not in labels, or feeling off-list)`,
  )
  console.log(
    `new emojis added    : ${newEmojis.length ? newEmojis.join(" ") : "none"}`,
  )
  if (droppedEmoji.length) {
    console.log(
      `new emojis rejected : ${droppedEmoji
        .map(([e, n]) => `${e}:${n}`)
        .join(" ")}`,
    )
  }
}

// -------------------------------------------------------------------- driver

if (import.meta.main) {
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  const feelings = z
    .object({ feelings: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8"))).feelings

  const texts = await step1()
  if (texts.length === 0) {
    console.log("no new unique texts this run; nothing to annotate")
    await rm(RAW, { force: true })
    process.exit(0)
  }

  const records = await step2(texts, feelings)
  await step3(records)

  await rm(RAW, { force: true })
  await rm(TMP, { force: true })
  console.log("\ndone. data.jsonl + labels.json updated; tmp files removed.")
  process.exit(0)
}
