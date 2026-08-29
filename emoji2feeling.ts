/**
 * Data pipeline, emoji-first entry point -- upsamples the rarest emojis.
 *
 *   bun emoji2feeling.ts  ->  bun gen_labels.ts
 *
 * One run does the whole thing, no intermediate file:
 *
 *   count every labels.json emoji's rows in data.jsonl, take the 100 rarest
 *     -> generate 20 short WhatsApp-style texts per emoji           (phase 1)
 *     -> feeling-annotate the fresh texts (closed labels.json set)  (phase 2)
 *     -> append {emoji, text, feeling} to data.jsonl
 *
 *   - emoji:   carried straight through from generation (this script owns it)
 *   - feeling: annotated, constrained to the labels.json feelings list
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

const RARE_EMOJI_COUNT = 100
const TEXTS_PER_EMOJI = 20
const GEN_CONCURRENCY = 20

const ANNOTATE_BATCH_SIZE = 10
const ANNOTATE_CONCURRENCY = 20

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

// Emotion-neutral speaker roles -- demographic / relationship only, for voice
// spread. The emoji, not the voice, drives what each message is about.
const VOICES = [
  "a teenager", "a college student", "a new parent", "a retiree",
  "a shift worker", "a freelancer", "someone in their 30s", "a grandparent",
  "an office worker", "a nurse", "a tradesperson", "a student athlete",
  "a girl", "a boy",
  "a father", "a mother", "a sibling", "a cousin", "a friend", "a neighbor",
  "a coworker", "a classmate", "a teammate", "a mentor", "a mentee",
]

type Row = { emoji: string; feeling: string; text: string }
type Candidate = { emoji: string; text: string }

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

function pickVoice(): string {
  return VOICES[Math.floor(Math.random() * VOICES.length)]
}

/**
 * Tally each labels.json emoji's rows in data.jsonl and return the `n` least
 * represented, rarest first. Palette emojis absent from data.jsonl count as 0;
 * ties keep labels.json order. Emojis outside the palette are ignored -- they
 * get dropped by gen_labels.ts / data.py anyway, so upsampling them is wasted.
 */
function rarestEmojis(rows: Row[], palette: string[], n: number): string[] {
  const counts = new Map<string, number>(palette.map((e) => [e, 0]))
  for (const r of rows) {
    const c = counts.get(r.emoji)
    if (c !== undefined) counts.set(r.emoji, c + 1)
  }
  const ranked = [...counts.entries()].sort((a, b) => a[1] - b[1])
  const picked = ranked.slice(0, n)
  console.log(
    `${counts.size} palette emojis -> upsampling ${picked.length} rarest ` +
    `(${picked[0]?.[1]}..${picked[picked.length - 1]?.[1]} rows each)`,
  )
  return picked.map(([e]) => e)
}

// --- phase 1: emoji-guided text generation --------------------------------
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
        .replace(/^\s*(?:[-*•]|\d+[.)])\s+/, "") // strip a list marker
        .replace(/^["'`]+|["'`]+$/g, "") // strip wrapping quotes/backticks
        .trim(),
    )
    .filter((l) => l && !l.startsWith("```"))
}

// --- phase 2: feeling annotation (closed set) --------------------------
const Annotation = z.object({ id: z.number(), feeling: z.string() })

/**
 * Annotate one batch with a feeling from `feelings`. Returns id -> feeling for
 * every id the model answered with a valid feeling. Makes up to two attempts (a
 * partial or failed response retries the whole batch); whatever is still
 * missing after that is logged and left out.
 */
async function annotateBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, string>> {
  const ids = new Set(batch.map((b) => b.id))
  const valid = new Set(feelings)
  const byId = new Map<number, string>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [
          "You are an annotator. For each message below, pick the single feeling that best fits it.",
          `Choose exactly one from this list: ${feelings.join(", ")}.`,
          "",
          "Return exactly one annotation object per input message, echoing its id.",
          "Do not add, drop, reorder, or merge items.",
          `Format: {"annotations": [{"id": 0, "feeling": "${feelings[0]}"}]}`,
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        const f = a.feeling.trim()
        if (ids.has(a.id) && valid.has(f)) byId.set(a.id, f)
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`\n  feeling batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (!byId.has(b.id)) console.warn(`\n  dropped id ${b.id}: no feeling`)
  }
  return byId
}

if (import.meta.main) {
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  if (!existsSync(DATA)) throw new Error(`${DATA} not found`)
  const labels = z
    .object({ feelings: z.array(z.string()), emojis: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8")))
  const feelings = labels.feelings
  const palette = labels.emojis

  const rows: Row[] = []
  for (const l of (await readFile(DATA, "utf8")).split("\n")) {
    const line = l.trim()
    if (line) rows.push(JSON.parse(line) as Row)
  }

  const targets = rarestEmojis(rows, palette, RARE_EMOJI_COUNT)

  // Texts already in the corpus -- never regenerate them.
  const inCorpus = new Set<string>()
  for (const r of rows) {
    const n = normalize(r.text)
    if (n) inCorpus.add(n)
  }
  console.log(`${inCorpus.size} existing texts to dedup against`)

  // --- phase 1: generate --------------------------------------------------
  const genBar = new cliProgress.SingleBar(
    {
      format:
        "generating |{bar}| {percentage}% | {value}/{total} emojis | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  genBar.start(targets.length, 0)

  const seen = new Set<string>()
  const candidates: Candidate[] = []

  const genQ = new PQueue({ concurrency: GEN_CONCURRENCY })
  genQ.addAll(
    targets.map((emoji) => async () => {
      try {
        for (const text of await genBatch(pickVoice(), emoji)) {
          const n = normalize(text)
          if (!n || inCorpus.has(n) || seen.has(n)) continue
          seen.add(n)
          candidates.push({ emoji, text })
        }
      } catch (err) {
        console.warn(`\n  gen batch (${emoji}) failed: ${err}`)
      }
      genBar.increment()
    }),
  )
  await genQ.onIdle()
  genBar.stop()

  // --- phase 2: annotate ------------------------------------------------
  const todo = candidates
    .filter((c) => normalize(c.text).length <= MAX_TEXT_LEN)
    .map((c, id) => ({ id, emoji: c.emoji, text: c.text }))
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
          feelings,
        )
        const out: string[] = []
        for (const b of batch) {
          const feeling = byId.get(b.id)
          if (!feeling) continue
          out.push(JSON.stringify({ emoji: b.emoji, text: b.text, feeling }))
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
