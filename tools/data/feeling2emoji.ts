/**
 * Data pipeline, feeling-first entry point (was feeling.ts -> emoji.ts).
 *
 *   bun feeling2emoji.ts  ->  bun gen_labels.ts
 *
 * One run does the whole thing, no intermediate file:
 *
 *   pick 3 rarest feelings
 *     -> generate short WhatsApp-style texts that convey them        (phase 1)
 *     -> annotate each fresh text with an emoji + a feeling          (phase 2)
 *     -> append {feeling, text, emoji}, dropping any row whose
 *        phase-2 feeling disagrees with the seed feeling
 *
 *   - feeling: seeded by generation, then kept only if the phase-2 annotator --
 *              judging the text cold against the closed labels.json list --
 *              independently lands on the same feeling; drift is dropped
 *   - emoji:   free choice, whatever the annotator picks
 *
 * Both phases are PQueue-parallelized. Nothing is retried across runs: a failed
 * generate or annotate batch is logged and dropped, and the next run just
 * regenerates. Coverage counting mirrors data.py's load-time filter (a row
 * counts only if its feeling and emoji are both in labels.json); the length
 * check below only decides what is worth an API call.
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
const GEN_BATCH_SIZE = 100
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

// Calibration for the phase-2 feeling check. Counters a measured bias (see
// data.md, 2026-08-29) toward treating flat, practical text as a strong feeling
// instead of Neutral -- here it decides which seeded rows survive.
const FEELING_GUIDANCE = [
  "Label the emotion a typical reader would actually feel from the words, not",
  "one that is merely plausible for the situation.",
  '- "Neutral" is the right answer for flat, practical or informational',
  "  messages: logistics, scheduling, quick factual updates, plain questions,",
  "  low-effort replies. Do not upgrade these to a stronger feeling.",
  '- Do not inflate. A caring or domestic line ("your socks are on the',
  '  radiator") is Neutral unless it openly states affection -- only then Love.',
  "  Mild irritation with no heat is Neutral, not Angry. A dry or self-mocking",
  "  complaint is Neutral, not Sad. Plainly stated anticipation is Neutral, not",
  "  Happy.",
  "- Reserve Love, Sad and Angry for messages where that feeling is",
  "  unmistakably on the surface.",
  "- If two feelings fit, pick the milder; if none clearly fits, pick Neutral.",
].join("\n")

// Phase-1 steer for how hard to lean on the seed feeling. Neutral is the one
// target that is *meant* to be affect-free, so it gets the opposite instruction.
function conveyInstruction(feeling: string): string[] {
  if (feeling === "Neutral") {
    return [
      "Every message must be genuinely emotion-free: a plain, practical or",
      "informational note (logistics, scheduling, a quick fact, a low-key",
      "question), with no detectable mood, positive or negative.",
    ]
  }
  return [
    `Every message must unmistakably convey the feeling "${feeling}" -- someone`,
    "reading it cold, with no context, should name that feeling. Convey it",
    "through what is said and how, never by naming the feeling.",
    "A flat or logistical message that a person in that mood merely could have",
    "sent does not count: the feeling has to be visible in the words.",
  ]
}

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
 *
 * Only rows data.py would keep are counted: a row contributes to its feeling's
 * tally only if both its feeling and its emoji are in labels.json. (The
 * text-length filter is not applied here.)
 */
function rarestFeelings(
  feelings: string[],
  emojis: Set<string>,
  rows: Row[],
  n: number,
): string[] {
  const counts = new Map<string, number>(feelings.map((f) => [f, 0]))
  for (const r of rows) {
    if (counts.has(r.feeling) && emojis.has(r.emoji)) {
      counts.set(r.feeling, counts.get(r.feeling)! + 1)
    }
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
      ...conveyInstruction(feeling),
      `One message per line. No numbering, no bullets, no quotes, no emoji, no commentary.`,
      `Each message at most ${MAX_RAW_LEN} characters.`,
      `Vary the wording, situation and sentence shape, but keep every line firmly in the target feeling -- do not drift toward a different mood for the sake of variety.`,
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

// --- phase 2: emoji annotation + feeling cross-check ---------------------
const Annotation = z.object({
  id: z.number(),
  emoji: z.string(),
  feeling: z.string(),
})

/**
 * Annotate one batch. Returns id -> {emoji, feeling} for every id the model
 * answered: `emoji` is a free pick, `feeling` is the annotator's own cold read
 * of the text against the closed list (the caller drops rows where it disagrees
 * with the seed feeling). Makes up to two attempts (a partial or failed
 * response retries the whole batch); whatever is still missing is logged.
 */
async function annotateBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, { emoji: string; feeling: string }>> {
  const ids = new Set(batch.map((b) => b.id))
  const byId = new Map<number, { emoji: string; feeling: string }>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
    try {
      const { output } = await generateText({
        model: MODEL,
        output: Output.object({
          schema: z.object({ annotations: z.array(Annotation) }),
        }),
        prompt: [
          "You are an annotator. For each message below, do two things:",
          "1. pick the single emoji that best fits it -- any emoji, the most fitting one, not a safe default.",
          `2. pick the single feeling that best fits it, exactly one from this list: ${feelings.join(", ")}.`,
          "",
          FEELING_GUIDANCE,
          "",
          "Return exactly one annotation object per input message, echoing its id.",
          "Do not add, drop, reorder, or merge items.",
          `Format: {"annotations": [{"id": 0, "emoji": "\u{1F600}", "feeling": "${feelings[0]}"}]}`,
          "",
          "Messages:",
          JSON.stringify(batch),
        ].join("\n"),
      })
      for (const a of output.annotations) {
        if (ids.has(a.id)) {
          byId.set(a.id, { emoji: a.emoji.trim(), feeling: a.feeling.trim() })
        }
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`\n  emoji batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (!byId.has(b.id)) console.warn(`\n  dropped id ${b.id}: no annotation`)
  }
  return byId
}

if (import.meta.main) {
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  const labels = z
    .object({ feelings: z.array(z.string()), emojis: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8")))
  const feelings = labels.feelings
  const emojis = new Set(labels.emojis)

  const rows: Row[] = []
  if (existsSync(DATA)) {
    for (const l of (await readFile(DATA, "utf8")).split("\n")) {
      const line = l.trim()
      if (line) rows.push(JSON.parse(line) as Row)
    }
  }
  const targets = rarestFeelings(feelings, emojis, rows, FEELINGS_PER_RUN)
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
  let drifted = 0
  const validFeelings = new Set(feelings)
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
          const rec = byId.get(b.id)
          if (!rec || !validFeelings.has(rec.feeling)) continue
          // Seed feeling stands only if a cold read of the text agrees with it.
          if (rec.feeling !== b.feeling) {
            drifted++
            continue
          }
          out.push(
            JSON.stringify({ feeling: b.feeling, text: b.text, emoji: rec.emoji }),
          )
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
  console.log(`dropped (feeling drift) : ${drifted}`)
  console.log(`annotated -> data.jsonl : ${annotated}`)
  console.log("\nnext: bun gen_labels.ts")
  process.exit(0)
}
