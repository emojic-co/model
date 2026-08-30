import { existsSync } from "node:fs"
import { appendFile, readFile, rename, writeFile } from "node:fs/promises"

import { generateText, Output } from "ai"
import cliProgress from "cli-progress"
import { z } from "zod"

const MODEL = "openai/gpt-5.6-luna"
const BATCH_SIZE = 10
const CONCURRENCY = 10
const MIN_COUNT = 20

const DATA = "./data.jsonl"
const DATA_BAK = "./data.jsonl.bak"
const DATA_TMP = "./data.jsonl.tmp"
const DATA_DRY = "./data.jsonl.dry"
const LABELS = "./labels.json"
const LABELS_BAK = "./labels.json.bak"

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

const Annotation = z.object({
  id: z.number(),
  feeling: z.string(),
  emoji: z.string(),
})

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
          console.warn(`  dropped id ${m.id}: no annotation returned`)
        }
        return byId
      }
    } catch (err) {
      if (attempt === 1) {
        console.warn(`  batch of ${batch.length} failed twice, skipped: ${err}`)
        return new Map()
      }
    }
  }
  return new Map()
}

function rebuildLabelList(
  counts: Map<string, number>,
  seed: string[],
): string[] {
  const kept = new Set(
    [...counts.entries()].filter(([, n]) => n >= MIN_COUNT).map(([k]) => k),
  )
  const seedSurvivors = seed.filter((s) => kept.has(s))
  const extras = [...kept]
    .filter((k) => !seedSurvivors.includes(k))
    .sort((a, b) => (counts.get(b) ?? 0) - (counts.get(a) ?? 0))
  return [...seedSurvivors, ...extras]
}

function countBy(rows: Row[], key: "emoji" | "feeling"): Map<string, number> {
  const m = new Map<string, number>()
  for (const r of rows) m.set(r[key], (m.get(r[key]) ?? 0) + 1)
  return m
}

if (import.meta.main) {
  const argv = process.argv.slice(2)
  const limitArg = argv.find((a) => a === "--limit" || a.startsWith("--limit="))
  let limit = Infinity
  if (limitArg) {
    const v = limitArg.includes("=")
      ? limitArg.split("=")[1]
      : argv[argv.indexOf(limitArg) + 1]
    limit = parseInt(v ?? "", 10)
    if (!Number.isFinite(limit) || limit <= 0) {
      throw new Error("--limit needs a positive integer")
    }
  }
  const DRY = Number.isFinite(limit)
  const resuming = existsSync(DATA_TMP)

  if (!DRY && !resuming && (existsSync(DATA_BAK) || existsSync(LABELS_BAK))) {
    throw new Error(
      `${DATA_BAK} / ${LABELS_BAK} already exist from a previous run. ` +
        "Restore or delete the backups before re-running (a partial run leaves " +
        "data.jsonl.tmp, which triggers resume automatically).",
    )
  }

  const seedPath = existsSync(LABELS) ? LABELS : LABELS_BAK
  const seed = z
    .object({ feelings: z.array(z.string()), emojis: z.array(z.string()) })
    .parse(JSON.parse(await readFile(seedPath, "utf8")))
  const feelings = seed.feelings

  const inputPath = resuming || DRY ? (existsSync(DATA) ? DATA : DATA_BAK) : DATA
  const rawRows = await readJsonl(existsSync(inputPath) ? inputPath : DATA_BAK)
  const seenNorm = new Set<string>()
  const uniqueTexts: { id: number; text: string }[] = []
  for (const r of rawRows) {
    const n = normalize(r.text)
    if (!n || seenNorm.has(n)) continue
    seenNorm.add(n)
    uniqueTexts.push({ id: uniqueTexts.length, text: r.text })
  }
  console.log(
    `${rawRows.length} rows -> ${uniqueTexts.length} unique normalized texts`,
  )

  if (DRY) {
    const sample = uniqueTexts.slice(0, limit)
    const out: Row[] = []
    const dryBatches = chunk(sample, BATCH_SIZE)
    const dryBar = new cliProgress.SingleBar(
      {
        format:
          "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
      },
      cliProgress.Presets.shades_classic,
    )
    dryBar.start(dryBatches.length, 0)
    for (const batch of dryBatches) {
      const byId = await annotateBatch(batch, feelings)
      for (const b of batch) {
        const a = byId.get(b.id)
        if (a) out.push({ emoji: a.emoji, feeling: a.feeling, text: b.text })
      }
      dryBar.increment()
    }
    dryBar.stop()
    await writeFile(DATA_DRY, out.map((r) => JSON.stringify(r)).join("\n") + "\n")
    const em = countBy(out, "emoji")
    const fe = countBy(out, "feeling")
    console.log(`\nwrote ${out.length}/${sample.length} -> ${DATA_DRY}`)
    console.log("feelings:", Object.fromEntries(fe))
    console.log("distinct emojis:", em.size, [...em.keys()].join(" "))
    console.log("\nfirst rows:")
    for (const r of out.slice(0, 10)) console.log(" ", JSON.stringify(r))
    process.exit(0)
  }

  if (!resuming) {
    await rename(DATA, DATA_BAK)
  }
  if (existsSync(LABELS) && !existsSync(LABELS_BAK)) {
    await rename(LABELS, LABELS_BAK)
  }

  const done = new Set<string>()
  if (existsSync(DATA_TMP)) {
    for (const r of await readJsonl(DATA_TMP)) done.add(normalize(r.text))
  }
  const todo = uniqueTexts.filter((u) => !done.has(normalize(u.text)))
  console.log(
    `${done.size} already annotated, ${todo.length} to go ` +
      `(${Math.ceil(todo.length / BATCH_SIZE)} batches)`,
  )

  const batches = chunk(todo, BATCH_SIZE)
  const bar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  bar.start(batches.length, 0)

  let writeChain: Promise<void> = Promise.resolve()
  const append = (text: string) => {
    writeChain = writeChain.then(() => appendFile(DATA_TMP, text))
    return writeChain
  }

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
    CONCURRENCY,
  )
  await writeChain
  bar.stop()

  await rename(DATA_TMP, DATA)

  const annotated = await readJsonl(DATA)
  const emojiList = rebuildLabelList(countBy(annotated, "emoji"), seed.emojis)
  const feelingList = rebuildLabelList(countBy(annotated, "feeling"), feelings)
  const keepEmoji = new Set(emojiList)
  const keepFeeling = new Set(feelingList)

  await writeFile(
    LABELS,
    JSON.stringify({ feelings: feelingList, emojis: emojiList }, null, 2) + "\n",
  )

  const filtered = annotated.filter(
    (r) => keepEmoji.has(r.emoji) && keepFeeling.has(r.feeling),
  )
  await writeFile(
    DATA,
    filtered.map((r) => JSON.stringify(r)).join("\n") + "\n",
  )

  const dropEmoji = [...countBy(annotated, "emoji").entries()]
    .filter(([e]) => !keepEmoji.has(e))
    .sort((a, b) => b[1] - a[1])
  const dropFeeling = [...countBy(annotated, "feeling").entries()].filter(
    ([f]) => !keepFeeling.has(f),
  )
  console.log("\n--- summary ---")
  console.log(`unique texts annotated : ${annotated.length}`)
  console.log(
    `emojis kept            : ${emojiList.length} (${emojiList.join(" ")})`,
  )
  console.log(
    `emojis dropped (<${MIN_COUNT})   : ${dropEmoji.length} ` +
      `[${dropEmoji.map(([e, n]) => `${e}:${n}`).join(" ")}]`,
  )
  console.log(
    `feelings kept          : ${feelingList.length} (${feelingList.join(", ")})`,
  )
  console.log(
    `feelings dropped (<${MIN_COUNT}) : ${dropFeeling
      .map(([f, n]) => `${f}:${n}`)
      .join(" ") || "none"}`,
  )
  console.log(
    `data.jsonl rows        : ${annotated.length} -> ${filtered.length} ` +
      `(${annotated.length - filtered.length} dropped by threshold)`,
  )
  console.log(
    "\nnext: review labels.json, ensure FEELING_PALETTE covers every kept " +
      "feeling, then `uv run main.py` (which also refreshes web/public/).",
  )
  process.exit(0)
}
