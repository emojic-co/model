/**
 * Data pipeline, feeling-first entry point (was feeling.ts -> emoji.ts).
 *
 *   bun feeling2emoji.ts  ->  bun gen_labels.ts
 *
 * One run does the whole thing, no intermediate file:
 *
 *   pick 3 rarest feelings
 *     -> generate short WhatsApp-style texts that convey them   (phase 1)
 *     -> emoji-annotate the fresh texts                          (phase 2)
 *     -> append {feeling, text, emoji} to data.jsonl
 *
 *   - feeling: carried straight through from generation (this script owns it)
 *   - emoji:   free choice, whatever the annotator picks
 *
 * Both phases are PQueue-parallelized. Nothing is retried across runs: a failed
 * generate or annotate batch is logged and dropped, and the next run just
 * regenerates. No label filtering happens here -- that is data.py's job at train
 * time; the length check below only decides what is worth an API call.
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { existsSync } from "node:fs"
import { appendFile, readFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const DATA = "./data.jsonl"
const LABELS = "./labels.json"

// Phase 1 keeps anything <= 50 chars; the tighter MAX_TEXT_LEN cut is a runtime
// concern, applied in phase 2 (and again by data.py), not during generation.
const MAX_RAW_LEN = 50
// Mirror of config.py's MAX_TEXT_LEN. A line whose normalized form is longer is
// not worth annotating -- data.py would filter the record at train time anyway.
const MAX_TEXT_LEN = 42

const FEELINGS_PER_RUN = 3
const BATCHES_PER_FEELING = 10
const GEN_BATCH_SIZE = 50
const GEN_CONCURRENCY = 10

const ANNOTATE_BATCH_SIZE = 10
const ANNOTATE_CONCURRENCY = 10

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
// feeling comes entirely from the prompt, not from the voice.
const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

type Row = { emoji: string; feeling: string; text: string }
type Candidate = { feeling: string; text: string }

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

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

// --- phase 1: feeling-guided text generation -------------------------------
async function genBatch(voice: string, feeling: string): Promise<string[]> {
  const { text } = await generateText({
    model: MODEL,
    prompt: [
      `Write ${GEN_BATCH_SIZE} short WhatsApp-style messages as if sent by ${voice}.`,
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

// --- phase 2: emoji annotation -------------------------------------------
const Annotation = z.object({ id: z.number(), emoji: z.string() })

/**
 * Annotate one batch. Returns id -> emoji for every id the model answered.
 * Makes up to two attempts (a partial or failed response retries the whole
 * batch); whatever is still missing after that is logged and left out.
 */
async function annotateBatch(
  batch: { id: number; text: string }[],
): Promise<Map<number, string>> {
  const ids = new Set(batch.map((b) => b.id))
  const byId = new Map<number, string>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [
          "You are an annotator. For each message below, pick the single emoji that best fits it.",
          "Any emoji is allowed - pick the most fitting one, not a safe default.",
          "",
          "Return exactly one annotation object per input message, echoing its id.",
          "Do not add, drop, reorder, or merge items.",
          'Format: {"annotations": [{"id": 0, "emoji": "\u{1F600}"}]}',
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        if (ids.has(a.id)) byId.set(a.id, a.emoji.trim())
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`\n  emoji batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (!byId.has(b.id)) console.warn(`\n  dropped id ${b.id}: no emoji`)
  }
  return byId
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
  const targets = rarestFeelings(feelings, rows, FEELINGS_PER_RUN)
  console.log(`target feelings (round-robin per batch): ${targets.join(", ")}`)

  // Texts already in the corpus -- never regenerate them.
  const inCorpus = new Set<string>()
  for (const r of rows) {
    const n = normalize(r.text)
    if (n) inCorpus.add(n)
  }
  console.log(`${inCorpus.size} existing texts to dedup against`)

  // --- phase 1: generate --------------------------------------------------
  const batchCount = targets.length * BATCHES_PER_FEELING
  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(batchCount, 0)

  const seen = new Set<string>()
  const candidates: Candidate[] = []

  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    Array.from({ length: batchCount }, (_, i) => async () => {
      const feeling = targets[i % targets.length]
      try {
        for (const text of await genBatch(pickVoice(), feeling)) {
          const n = normalize(text)
          if (!n || inCorpus.has(n) || seen.has(n)) continue
          seen.add(n)
          candidates.push({ feeling, text })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${feeling}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()

  // --- phase 2: annotate ------------------------------------------------
  const todo = candidates
    .filter((c) => normalize(c.text).length <= MAX_TEXT_LEN)
    .map((c, id) => ({ id, feeling: c.feeling, text: c.text }))
  const nLong = candidates.length - todo.length
  console.log(
    `\n${candidates.length} fresh texts (${nLong} too long, ${todo.length} to annotate)`,
  )

  let annotated = 0
  if (todo.length) {
    const batches = chunk(todo, ANNOTATE_BATCH_SIZE)
    const annBar = new cliProgress.SingleBar(
      {
        format:
          "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
      },
      cliProgress.Presets.shades_classic,
    )
    annBar.start(batches.length, 0)

    // Serialize appends so concurrent workers don't interleave partial lines.
    let writeChain: Promise<unknown> = Promise.resolve()
    const append = (s: string) => {
      writeChain = writeChain.then(() => appendFile(DATA, s))
      return writeChain
    }

    const annQ = new PQueue({ concurrency: ANNOTATE_CONCURRENCY })
    annQ.addAll(
      batches.map((batch) => async () => {
        const byId = await annotateBatch(
          batch.map((b) => ({ id: b.id, text: b.text })),
        )
        const out: string[] = []
        for (const b of batch) {
          const emoji = byId.get(b.id)
          if (!emoji) continue
          out.push(JSON.stringify({ feeling: b.feeling, text: b.text, emoji }))
        }
        if (out.length) {
          annotated += out.length
          await append(out.join("\n") + "\n")
        }
        annBar.increment()
      }),
    )
    await annQ.onIdle()
    await writeChain
    annBar.stop()
  }

  console.log("\n--- summary ---")
  console.log(`generated fresh texts   : ${candidates.length}`)
  console.log(`skipped (too long)      : ${nLong}`)
  console.log(`annotated -> data.jsonl : ${annotated}`)
  console.log("\nnext: bun gen_labels.ts")
  process.exit(0)
}
