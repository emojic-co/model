import { DATA_JSONL } from "../../files.ts"
import { readJsonl } from "../data/io.ts"
import { normalize } from "../data/normalize.ts"

type Row = { text?: unknown }

export function countDistinctNormalized(rows: Row[]): {
  rows: number
  withText: number
  nonEmpty: number
  distinct: number
} {
  const seen = new Set<string>()
  let withText = 0
  let nonEmpty = 0
  for (const row of rows) {
    if (typeof row.text !== "string") continue
    withText++
    const norm = normalize(row.text)
    if (!norm) continue
    nonEmpty++
    seen.add(norm)
  }
  return { rows: rows.length, withText, nonEmpty, distinct: seen.size }
}

if (import.meta.main) {
  const rows = await readJsonl<Row>(DATA_JSONL)
  const s = countDistinctNormalized(rows)
  console.log(`source            : ${DATA_JSONL}`)
  console.log(`rows              : ${s.rows}`)
  console.log(`rows with text    : ${s.withText}`)
  console.log(`non-empty normalized : ${s.nonEmpty}`)
  console.log(`distinct normalized  : ${s.distinct}`)
}
