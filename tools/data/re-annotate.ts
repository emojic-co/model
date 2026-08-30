import { existsSync } from "node:fs"
import { readFile } from "node:fs/promises"

import cliProgress from "cli-progress"

import { annotate, annotateBatchCount } from "./annotate.ts"
import { appendJsonl, writeFileAtomic } from "./io.ts"

const DATA = "./data.jsonl"
const TRAIN = "./train.jsonl"

const SAMPLE_SIZE = 10_000

function sampleIndices(n: number, k: number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i)
  const take = Math.min(k, n)
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(Math.random() * (n - i))
      ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  return idx.slice(0, take)
}

if (import.meta.main) {
  if (!existsSync(DATA)) {
    console.log(`${DATA} not found, nothing to re-annotate`)
    process.exit(0)
  }

  const lines = (await readFile(DATA, "utf8")).split("\n").filter((l) => l.trim())
  if (!lines.length) {
    console.log(`${DATA} is empty, nothing to re-annotate`)
    process.exit(0)
  }

  const picked = sampleIndices(lines.length, SAMPLE_SIZE)
  const source: number[] = []
  const texts: string[] = []
  let malformed = 0
  for (const i of picked) {
    let text: unknown
    try {
      text = JSON.parse(lines[i]).text
    } catch {
      malformed++
      continue
    }
    if (typeof text !== "string" || !text) {
      malformed++
      continue
    }
    source.push(i)
    texts.push(text)
  }
  console.log(`re-annotating ${texts.length} rows from ${DATA}`)

  const annBar = new cliProgress.SingleBar(
    {
      format:
        "annotating |{bar}| {percentage}% | {value}/{total} batches | ETA: {eta}s",
    },
    cliProgress.Presets.shades_classic,
  )
  annBar.start(annotateBatchCount(texts.length), 0)
  const labels = await annotate(texts, () => annBar.increment())
  annBar.stop()

  const appended: string[] = []
  const consumed = new Set<number>()
  for (let p = 0; p < texts.length; p++) {
    const label = labels.get(p)
    if (!label) continue
    appended.push(
      JSON.stringify({
        text: texts[p],
        feeling: label.feeling,
        emoji: label.emoji,
      }),
    )
    consumed.add(source[p])
  }

  await appendJsonl(TRAIN, appended)

  const kept = lines.filter((_, i) => !consumed.has(i))
  await writeFileAtomic(DATA, kept.length ? kept.join("\n") + "\n" : "", true)

  console.log("\n--- summary ---")
  console.log(`sampled              : ${picked.length}`)
  console.log(`malformed (skipped)  : ${malformed}`)
  console.log(`re-annotated -> train: ${appended.length}`)
  console.log(`left in data.jsonl   : ${kept.length} (was ${lines.length})`)
  process.exit(0)
}
