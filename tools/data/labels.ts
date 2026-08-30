import { existsSync } from "node:fs"
import { readFile, writeFile } from "node:fs/promises"
import { TOP_EMOJIS, TOP_FEELINGS } from "./config"

const TRAIN = "./train.jsonl"
const LABELS = "./labels.json"

function topN(counts: Map<string, number>, n: number): string[] {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k)
}

if (import.meta.main) {
  if (!existsSync(TRAIN)) throw new Error(`${TRAIN} not found`)

  const emojis = new Map<string, number>()
  const feelings = new Map<string, number>()
  for (const line of (await readFile(TRAIN, "utf8")).split("\n")) {
    if (!line.trim()) continue
    const row = JSON.parse(line)
    emojis.set(row.emoji, (emojis.get(row.emoji) ?? 0) + 1)
    feelings.set(row.feeling, (feelings.get(row.feeling) ?? 0) + 1)
  }

  const out = {
    feelings: topN(feelings, TOP_FEELINGS),
    emojis: topN(emojis, TOP_EMOJIS),
  }
  await writeFile(LABELS, JSON.stringify(out, null, 2) + "\n")

  console.log(
    `wrote ${LABELS}: ${out.feelings.length} feelings, ${out.emojis.length} emojis`,
  )
  console.log(`  feelings: ${out.feelings.join(" ")}`)
  console.log(`  distinct in train: ${feelings.size} feelings, ${emojis.size} emojis`)
  process.exit(0)
}
