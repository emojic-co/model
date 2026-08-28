/**
 * Step 1 of the data pipeline: generate raw, unlabelled text.
 *
 *   bun raw_txt.ts  ->  bun annotation.ts  ->  bun gen_labels.ts
 *
 * Writes short, informal, WhatsApp-style messages across a wide spread of
 * topics and speaker voices and *appends* the new ones to raw.txt. Lines are
 * deduplicated -- against each other and against whatever raw.txt already
 * holds -- using the same normalized key as data.py. raw.txt is never
 * truncated or thrown away; annotation.ts is what drains it.
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { existsSync } from "node:fs"
import { appendFile, readFile } from "node:fs/promises"

import { generateText } from "ai"
import cliProgress from "cli-progress"

const MODEL = "openai/gpt-5.6-luna"
const RAW = "./raw.txt"

// Step 1 keeps anything <= 50 chars; the tighter MAX_TEXT_LEN cut is a
// runtime concern, applied later by annotation.ts / data.py, not here.
const MAX_RAW_LEN = 50

const TARGET_TEXTS = 1000
const GEN_BATCH = 25
const GEN_CONCURRENCY = 10

// --- normalized dedup key: mirror of data.py's normalize() ------------------
const VOCAB = new Set("abcdefghijklmnopqrstuvwxyz!?:()@$%&* ")
function normalize(text: string): string {
  const t = text
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/(.)\1{2,}/g, "$1$1")
  return [...t].filter((c) => VOCAB.has(c)).join("")
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
      `Each message at most ${MAX_RAW_LEN} characters.`,
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

if (import.meta.main) {
  const existing = new Set<string>()
  if (existsSync(RAW)) {
    for (const l of (await readFile(RAW, "utf8")).split("\n")) {
      const n = normalize(l)
      if (n) existing.add(n)
    }
    console.log(`raw.txt present -> ${existing.size} existing texts to dedup against`)
  }

  const nBatches = Math.ceil(TARGET_TEXTS / GEN_BATCH)
  const jobs = Array.from({ length: nBatches }, () => ({
    topic: TOPICS[Math.floor(Math.random() * TOPICS.length)],
    voice: VOICES[Math.floor(Math.random() * VOICES.length)],
  }))

  const seen = new Set<string>()
  const kept: string[] = []
  const bar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(nBatches, 0)
  await pMap(
    jobs,
    async (job) => {
      try {
        for (const line of await genBatch(job.topic, job.voice)) {
          if (line.length > MAX_RAW_LEN) continue
          const n = normalize(line)
          if (!n || existing.has(n) || seen.has(n)) continue
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

  if (kept.length) {
    await appendFile(RAW, kept.join("\n") + "\n")
  }
  console.log(`step 1: ${kept.length} unique new texts appended -> ${RAW}`)
  process.exit(0)
}
