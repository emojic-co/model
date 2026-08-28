/**
 * Step 2 of the data pipeline: label the raw text.
 *
 *   bun raw_txt.ts  ->  bun annotation.ts  ->  bun gen_labels.ts
 *
 * Drains raw.txt: every line is read, the ones worth annotating are sent to the
 * model in batches, and the merged {emoji, feeling, text} records are appended
 * to data.jsonl. Lines that were handled -- annotated, too long, or already in
 * data.jsonl -- are removed from raw.txt; lines whose annotation call failed
 * stay so the next run retries them. data.jsonl is only ever appended to.
 *
 *   - emoji:   free choice, whatever the model picks
 *   - feeling: exactly one from labels.json (a fixed closed set)
 *
 * No label or length filtering of the corpus happens here -- that is data.py's
 * job at train time. The length check below only decides what is worth an API
 * call now.
 *
 * Run:
 *   bun run annotation.ts
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
const LABELS = "./labels.json"
const RAW = "./raw.txt"

// Mirror of config.py's MAX_TEXT_LEN. A raw line whose normalized form is
// longer is not worth annotating -- data.py would filter the record at train
// time anyway.
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

const Annotation = z.object({
  id: z.number(),
  feeling: z.string(),
  emoji: z.string(),
})

type Label = { feeling: string; emoji: string }

/**
 * Annotate one batch. Returns id -> {feeling, emoji} for every id the model
 * answered. Makes up to two attempts (a partial or failed response triggers a
 * retry of the whole batch); whatever is still missing after that is logged
 * and left out.
 */
async function annotateBatch(
  batch: { id: number; text: string }[],
  feelings: string[],
): Promise<Map<number, Label>> {
  const ids = new Set(batch.map((b) => b.id))
  const byId = new Map<number, Label>()

  for (let attempt = 0; attempt < 2 && byId.size < ids.size; attempt++) {
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

      for (const a of output.annotations) {
        if (ids.has(a.id)) {
          byId.set(a.id, { feeling: a.feeling.trim(), emoji: a.emoji.trim() })
        }
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
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  const feelings = z
    .object({ feelings: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8"))).feelings

  if (!existsSync(RAW)) {
    console.log(`${RAW} not found -- run \`bun raw_txt.ts\` first`)
    process.exit(0)
  }

  // Raw queue, deduped against itself by normalized key (order preserved).
  const rawLines: string[] = []
  const rawSeen = new Set<string>()
  for (const l of (await readFile(RAW, "utf8")).split("\n")) {
    const line = l.trim()
    if (!line) continue
    const n = normalize(line)
    if (!n || rawSeen.has(n)) continue
    rawSeen.add(n)
    rawLines.push(line)
  }

  // Texts already in the corpus -- don't re-annotate them.
  const inCorpus = new Set<string>()
  if (existsSync(DATA)) {
    for (const r of await readJsonl(DATA)) inCorpus.add(normalize(r.text))
  }

  // Split rawLines into what's worth an API call (`todo`) and what's already
  // resolved (`handled`: too long or already in the corpus). Annotated `todo`
  // rows are added to `handled` below; whatever stays out of it stays in raw.txt.
  const todo: { id: number; text: string }[] = []
  const handled = new Set<string>()
  let nLong = 0
  let nDup = 0
  for (const text of rawLines) {
    const n = normalize(text)
    if (n.length > MAX_TEXT_LEN) {
      nLong++
      handled.add(n)
    } else if (inCorpus.has(n)) {
      nDup++
      handled.add(n)
    } else {
      todo.push({ id: todo.length, text })
    }
  }
  console.log(
    `raw.txt: ${rawLines.length} unique lines ` +
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
        const byId = await annotateBatch(batch, feelings)
        const rows: string[] = []
        for (const b of batch) {
          const a = byId.get(b.id)
          if (!a) continue
          rows.push(
            JSON.stringify({ emoji: a.emoji, feeling: a.feeling, text: b.text }),
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

  // Rewrite raw.txt without the handled lines; drop the file if nothing's left.
  const remaining = rawLines.filter((t) => !handled.has(normalize(t)))
  if (remaining.length) {
    await writeFile(RAW, remaining.join("\n") + "\n")
  } else {
    await rm(RAW, { force: true })
  }

  const annotated = handled.size - nLong - nDup
  console.log("\n--- summary ---")
  console.log(`annotated -> data.jsonl : ${annotated}`)
  console.log(`skipped (too long)      : ${nLong}`)
  console.log(`skipped (duplicate)     : ${nDup}`)
  console.log(
    `left in raw.txt (retry) : ${remaining.length}` +
    (remaining.length ? "" : " -- raw.txt removed"),
  )
  console.log("\nnext: bun gen_labels.ts")
  process.exit(0)
}
