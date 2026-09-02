import { existsSync } from "node:fs"
import { writeFile } from "node:fs/promises"

import { STYLES, TOP_EMOJIS } from "./config"
import { splitEmojis } from "./emoji.ts"
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
  for (const row of await readJsonl<{ emojis?: string; emoji?: string }>(TRAIN)) {
    const list =
      typeof row.emojis === "string"
        ? splitEmojis(row.emojis)
        : typeof row.emoji === "string"
          ? [row.emoji]
          : []
    for (const e of new Set(list)) emojis.set(e, (emojis.get(e) ?? 0) + 1)
  }

  const out = {
    styles: [...STYLES],
    emojis: topN(emojis, TOP_EMOJIS),
  }
  await writeFile(LABELS, JSON.stringify(out, null, 2) + "\n")

  console.log(
    `wrote ${LABELS}: ${out.styles.length} styles, ${out.emojis.length} emojis`,
  )
  console.log(`  distinct emojis in train: ${emojis.size}`)
  process.exit(0)
}
