/**
 * Step 1 of the data pipeline: feeling-guided text generation.
 *
 *   bun feeling.ts  ->  bun emoji.ts  ->  bun gen_labels.ts
 *
 * Reads data.jsonl, finds the 3 feelings from labels.json with the least
 * coverage, and generates short, informal, WhatsApp-style messages that convey
 * them -- one feeling per batch, round-robin across the 3, over a wide spread of
 * speaker voices. The new {feeling, text} records are *appended* to
 * feeling.jsonl. Lines are deduplicated -- against each other and against
 * whatever feeling.jsonl already holds -- using the same normalized key as
 * data.py. feeling.jsonl is never truncated or thrown away; emoji.ts is what
 * drains it.
 *
 * Repeated runs keep targeting whatever 3 are now rarest, so the corpus
 * rebalances over time.
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { existsSync } from "node:fs"
import { appendFile, readFile } from "node:fs/promises"

import { generateText } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const DATA = "./data.jsonl"
const LABELS = "./labels.json"
const FEELING_JSONL = "./feeling.jsonl"

// Step 1 keeps anything <= 50 chars; the tighter MAX_TEXT_LEN cut is a
// runtime concern, applied later by emoji.ts / data.py, not here.
const MAX_RAW_LEN = 50

const BATCH_COUNT = 50
const BATCH_SIZE = 50
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

// Emotion-neutral speaker roles -- demographic / relationship only, so the
// feeling comes entirely from the prompt below, not from the voice.
const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

type Row = { emoji: string; feeling: string; text: string }

/**
 * Tally the labels.json feelings across data.jsonl and return the `n` with the
 * fewest rows, least-covered first (ties: the order feelings appear in
 * labels.json). Feelings with no rows at all still rank first.
 */
function rarestFeelings(feelings: string[], rows: Row[], n: number): string[] {
  const counts = new Map<string, number>(feelings.map((f) => [f, 0]))
  for (const r of rows) {
    if (counts.has(r.feeling)) counts.set(r.feeling, counts.get(r.feeling)! + 1)
  }
  console.log(
    "feeling coverage in data.jsonl: " +
      feelings.map((f) => `${f}=${counts.get(f)}`).join(" "),
  )
  return feelings
    .map((f, i) => ({ f, i }))
    .sort((a, b) => counts.get(a.f)! - counts.get(b.f)! || a.i - b.i)
    .slice(0, n)
    .map((x) => x.f)
}

async function genBatch(voice: string, feeling: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
      `Every message must genuinely convey the feeling "${feeling}" -- through what is said and how, never by naming the feeling.`,
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary tone and intent: quick updates, dry humor, complaints, questions, sudden news, invitations, low-effort replies.`,
      `Sound real and specific.`,
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
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  const feelings = z
    .object({ feelings: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8"))).feelings

  const rows: Row[] = []
  if (existsSync(DATA)) {
    for (const l of (await readFile(DATA, "utf8")).split("\n")) {
      const line = l.trim()
      if (line) rows.push(JSON.parse(line) as Row)
    }
  }
  const targets = rarestFeelings(feelings, rows, 3)
  console.log(`target feelings (round-robin per batch): ${targets.join(", ")}`)

  const existing = new Set<string>()
  if (existsSync(FEELING_JSONL)) {
    for (const l of (await readFile(FEELING_JSONL, "utf8")).split("\n")) {
      const line = l.trim()
      if (!line) continue
      const n = normalize((JSON.parse(line) as { text: string }).text)
      if (n) existing.add(n)
    }
    console.log(
      `feeling.jsonl present -> ${existing.size} existing texts to dedup against`,
    )
  }

  const voices = Array.from(
    { length: BATCH_COUNT },
    () => VOICES[Math.floor(Math.random() * VOICES.length)],
  )

  const seen = new Set<string>()
  const bar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(BATCH_COUNT, 0)

  // Serialize appends so concurrent workers don't interleave partial lines.
  let writeChain: Promise<unknown> = Promise.resolve()
  const append = (s: string) => {
    writeChain = writeChain.then(() => appendFile(FEELING_JSONL, s))
    return writeChain
  }

  const q = new PQueue({ concurrency: GEN_CONCURRENCY })
  q.addAll(voices.map((voice, i) => async () => {
    const feeling = targets[i % targets.length]
    try {
      const batch = await genBatch(voice, feeling)
      const fresh: string[] = []
      for (const text of batch) {
        const n = normalize(text)
        if (!n || existing.has(n) || seen.has(n)) continue
        seen.add(n)
        fresh.push(JSON.stringify({ feeling, text }))
      }
      if (fresh.length) await append(fresh.join("\n") + "\n")
    } catch (err) {
      console.warn(`\n  gen batch (${voice}, ${feeling}) failed: ${err}`)
    }
    bar.increment()
  }))

  await q.onIdle()
  await writeChain
  bar.stop()

  console.log(`\nappended ${seen.size} new texts -> feeling.jsonl`)
  console.log("next: bun emoji.ts")
  process.exit(0)
}
