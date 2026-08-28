/**
 * Step 2 of the data pipeline: emoji-annotate the feeling-guided text.
 *
 *   bun feeling.ts  ->  bun emoji.ts  ->  bun gen_labels.ts
 *
 * Drains feeling.jsonl: every {feeling, text} line is read, the ones worth
 * annotating are sent to the model in batches, and the merged
 * {feeling, text, emoji} records are appended to data.jsonl. Lines that were
 * handled -- annotated, too long, or already in data.jsonl -- are removed from
 * feeling.jsonl; lines whose annotation call failed stay so the next run retries
 * them. data.jsonl is only ever appended to.
 *
 *   - feeling: carried straight through from feeling.jsonl (feeling.ts owns it)
 *   - emoji:   free choice, whatever the model picks
 *
 * No label or length filtering of the corpus happens here -- that is data.py's
 * job at train time. The length check below only decides what is worth an API
 * call now.
 *
 * Run:
 *   bun run emoji.ts
 *
 * Requires AI_GATEWAY_API_KEY (Bun auto-loads it from .env).
 */
import { existsSync } from "node:fs"
import { appendFile, readFile, rm, writeFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import PQueue from "p-queue"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const DATA = "./data.jsonl"
const FEELING_JSONL = "./feeling.jsonl"

// Mirror of config.py's MAX_TEXT_LEN. A line whose normalized form is longer is
// not worth annotating -- data.py would filter the record at train time anyway.
const MAX_TEXT_LEN = 42

const BATCH_SIZE = 10
const CONCURRENCY = 10

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

type Row = { emoji: string; feeling: string; text: string }
type Pending = { feeling: string; text: string }

function chunk<T>(arr: T[], n: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n))
  return out
}

const Annotation = z.object({
  id: z.number(),
  emoji: z.string(),
})

/**
 * Annotate one batch. Returns id -> emoji for every id the model answered.
 * Makes up to two attempts (a partial or failed response triggers a retry of
 * the whole batch); whatever is still missing after that is logged and left out.
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
        console.warn(`\n  batch of ${batch.length} failed: ${err}`)
      }
    }
  }

  for (const b of batch) {
    if (!byId.has(b.id)) console.warn(`\n  dropped id ${b.id}: no annotation`)
  }
  return byId
}

if (import.meta.main) {
  if (!existsSync(FEELING_JSONL)) {
    console.log(`${FEELING_JSONL} not found -- run \`bun feeling.ts\` first`)
    process.exit(0)
  }

  // Pending queue, deduped against itself by normalized text (order preserved).
  const pending: Pending[] = []
  const pendingSeen = new Set<string>()
  for (const l of (await readFile(FEELING_JSONL, "utf8")).split("\n")) {
    const line = l.trim()
    if (!line) continue
    const rec = JSON.parse(line) as Pending
    const n = normalize(rec.text)
    if (!n || pendingSeen.has(n)) continue
    pendingSeen.add(n)
    pending.push(rec)
  }

  // Texts already in the corpus -- don't re-annotate them.
  const inCorpus = new Set<string>()
  if (existsSync(DATA)) {
    for (const l of (await readFile(DATA, "utf8")).split("\n")) {
      const line = l.trim()
      if (line) inCorpus.add(normalize((JSON.parse(line) as Row).text))
    }
  }

  // Split pending into what's worth an API call (`todo`) and what's already
  // resolved (`handled`: too long or already in the corpus). Annotated `todo`
  // rows are added to `handled` below; whatever stays out of it stays in
  // feeling.jsonl.
  const todo: { id: number; feeling: string; text: string }[] = []
  const handled = new Set<string>()
  let nLong = 0
  let nDup = 0
  for (const rec of pending) {
    const n = normalize(rec.text)
    if (n.length > MAX_TEXT_LEN) {
      nLong++
      handled.add(n)
    } else if (inCorpus.has(n)) {
      nDup++
      handled.add(n)
    } else {
      todo.push({ id: todo.length, feeling: rec.feeling, text: rec.text })
    }
  }
  console.log(
    `feeling.jsonl: ${pending.length} unique lines ` +
    `(${nLong} too long, ${nDup} already in data.jsonl, ${todo.length} to annotate)`,
  )

  if (todo.length) {
    const batches = chunk(todo, BATCH_SIZE)
    const bar = new cliProgress.SingleBar(
      {
        format:
          "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
      },
      cliProgress.Presets.shades_classic,
    )
    bar.start(batches.length, 0)

    // Serialize appends so concurrent workers don't interleave partial lines.
    let writeChain: Promise<unknown> = Promise.resolve()
    const append = (s: string) => {
      writeChain = writeChain.then(() => appendFile(DATA, s))
      return writeChain
    }

    const q = new PQueue({ concurrency: CONCURRENCY })
    q.addAll(
      batches.map((batch) => async () => {
        const byId = await annotateBatch(
          batch.map((b) => ({ id: b.id, text: b.text })),
        )
        const rows: string[] = []
        for (const b of batch) {
          const emoji = byId.get(b.id)
          if (!emoji) continue
          rows.push(
            JSON.stringify({ feeling: b.feeling, text: b.text, emoji }),
          )
          handled.add(normalize(b.text))
        }
        if (rows.length) await append(rows.join("\n") + "\n")
        bar.increment()
      }),
    )

    await q.onIdle()
    await writeChain
    bar.stop()
  }

  // Rewrite feeling.jsonl without the handled lines; drop it if nothing's left.
  const remaining = pending.filter((r) => !handled.has(normalize(r.text)))
  if (remaining.length) {
    await writeFile(
      FEELING_JSONL,
      remaining.map((r) => JSON.stringify(r)).join("\n") + "\n",
    )
  } else {
    await rm(FEELING_JSONL, { force: true })
  }

  const annotated = handled.size - nLong - nDup
  console.log("\n--- summary ---")
  console.log(`annotated -> data.jsonl      : ${annotated}`)
  console.log(`skipped (too long)           : ${nLong}`)
  console.log(`skipped (duplicate)          : ${nDup}`)
  console.log(
    `left in feeling.jsonl (retry): ${remaining.length}` +
    (remaining.length ? "" : " -- feeling.jsonl removed"),
  )
  console.log("\nnext: bun gen_labels.ts")
  process.exit(0)
}
