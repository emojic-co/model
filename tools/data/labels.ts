import { existsSync } from "node:fs"
import { writeFile } from "node:fs/promises"
import { EKMAN_FEELINGS, TOP_EMOJIS } from "./config"
import { readJsonl } from "./io.ts"

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
  for (const row of await readJsonl<{ emoji: string }>(TRAIN)) {
    emojis.set(row.emoji, (emojis.get(row.emoji) ?? 0) + 1)
  }

  const out = {
    feelings: [...EKMAN_FEELINGS],
    emojis: topN(emojis, TOP_EMOJIS),
  }
  await writeFile(LABELS, JSON.stringify(out, null, 2) + "\n")

  console.log(
    `wrote ${LABELS}: ${out.feelings.length} feelings, ${out.emojis.length} emojis`,
  )
  console.log(`  feelings: ${out.feelings.join(" ")}`)
  console.log(`  distinct emojis in train: ${emojis.size}`)
  process.exit(0)
}
