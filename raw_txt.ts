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
import PQueue from "p-queue"

const MODEL = "openai/gpt-5.6-luna"
const RAW = "./raw.txt"

// Step 1 keeps anything <= 50 chars; the tighter MAX_TEXT_LEN cut is a
// runtime concern, applied later by annotation.ts / data.py, not here.
const MAX_RAW_LEN = 50

const BATCH_COUNT = 20
const BATCH_SIZE = 100
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

const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
  "someone from a different country", "someone from a different culture",
  "someone with a disability", "someone with a chronic illness",
  "someone who is introverted", "someone who is extroverted",
  "someone who is optimistic", "someone who is pessimistic",
  "someone who is sarcastic", "someone who is sincere",
  "someone who is humorous", "someone who is serious",
  "someone who is adventurous", "someone who is cautious",
  "someone who is spontaneous", "someone who is organized",
]

async function genBatch(voice: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
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
  const existing = new Set<string>()
  if (existsSync(RAW)) {
    for (const l of (await readFile(RAW, "utf8")).split("\n")) {
      const n = normalize(l)
      if (n) existing.add(n)
    }
    console.log(`raw.txt present -> ${existing.size} existing texts to dedup against`)
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
  const q = new PQueue({ concurrency: GEN_CONCURRENCY })
  q.addAll(voices.map((voice) => async () => {
    try {
      const batch = await genBatch(voice)
      await appendFile(RAW, batch.join("\n") + "\n")
    } catch (err) {
      console.warn(`\n  gen batch (${voice}) failed: ${err}`)
    }
    bar.increment()
  }))

  await q.onIdle()
  bar.stop()

  process.exit(0)
}
