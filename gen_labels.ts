/**
 * Step 3 of the data pipeline: (re)generate labels.json from data.jsonl.
 *
 *   bun raw_txt.ts  ->  bun annotation.ts  ->  bun gen_labels.ts
 *
 *   - feelings: a fixed closed set. Copied through from the existing
 *     labels.json untouched -- this script never adds, drops, or reorders one.
 *   - emojis:   every emoji in data.jsonl tallied, sorted by frequency
 *     (descending; ties keep first-seen order), and the top 100 written back
 *     in that order.
 *
 * data.jsonl is only read. Records whose emoji falls outside the top 100 stay
 * in the corpus -- data.py filters them out at train time.
 *
 * Run:
 *   bun run gen_labels.ts
 */
import { existsSync } from "node:fs"
import { readFile, writeFile } from "node:fs/promises"

import { z } from "zod"

const DATA = "./data.jsonl"
const LABELS = "./labels.json"
const TOP_N = 100

type Row = { emoji: string; feeling: string; text: string }

if (import.meta.main) {
  if (!existsSync(LABELS)) throw new Error(`${LABELS} not found`)
  if (!existsSync(DATA)) throw new Error(`${DATA} not found`)

  // feelings are fixed -- carry them through verbatim.
  const feelings = z
    .object({ feelings: z.array(z.string()) })
    .parse(JSON.parse(await readFile(LABELS, "utf8"))).feelings

  // tally emoji frequency; Map keeps first-seen order for tie-breaking.
  const counts = new Map<string, number>()
  for (const line of (await readFile(DATA, "utf8")).split("\n")) {
    const l = line.trim()
    if (!l) continue
    const { emoji } = JSON.parse(l) as Row
    counts.set(emoji, (counts.get(emoji) ?? 0) + 1)
  }

  // stable sort: equal counts keep insertion (first-seen) order.
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const emojis = ranked.slice(0, TOP_N).map(([e]) => e)

  await writeFile(
    LABELS,
    JSON.stringify({ feelings, emojis }, null, 2) + "\n",
  )

  const cutoff = ranked[TOP_N - 1]?.[1]
  console.log(
    `${counts.size} distinct emojis in data.jsonl -> kept top ${emojis.length}` +
      (cutoff !== undefined ? ` (>= ${cutoff} occurrences)` : ""),
  )
  const dropped = ranked.slice(TOP_N)
  if (dropped.length) {
    console.log(
      `dropped ${dropped.length}: ` +
        dropped
          .slice(0, 20)
          .map(([e, n]) => `${e}:${n}`)
          .join(" ") +
        (dropped.length > 20 ? " ..." : ""),
    )
  }
  process.exit(0)
}
