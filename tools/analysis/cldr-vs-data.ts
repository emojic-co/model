import { readFile } from "node:fs/promises"

import { DATA_JSONL } from "../../files.ts"
import { cldrEmojis } from "../data/cldr.ts"

const EXAMPLE_COUNT = 10

type Row = { text: string; emojis: string }
type Example = { text: string; cldrOnly: string[]; both: string[]; dataOnly: string[] }

function pickRandomIndices(total: number, count: number): Set<number> {
  const picked = new Set<number>()
  while (picked.size < Math.min(count, total)) {
    picked.add(Math.floor(Math.random() * total))
  }
  return picked
}

if (import.meta.main) {
  const lines = (await readFile(DATA_JSONL, "utf8")).split("\n").filter(Boolean)
  const sampleIndices = pickRandomIndices(lines.length, EXAMPLE_COUNT)

  let cldrOnlyTotal = 0
  let bothTotal = 0
  let dataOnlyTotal = 0
  const examples: Example[] = []

  lines.forEach((line, i) => {
    const row = JSON.parse(line) as Row
    const actual = new Set(row.emojis.trim() ? row.emojis.trim().split(/\s+/) : [])
    const predicted = new Set(actual.size ? cldrEmojis(row.text, actual.size) : [])

    const cldrOnly = [...predicted].filter((e) => !actual.has(e))
    const both = [...predicted].filter((e) => actual.has(e))
    const dataOnly = [...actual].filter((e) => !predicted.has(e))

    cldrOnlyTotal += cldrOnly.length
    bothTotal += both.length
    dataOnlyTotal += dataOnly.length

    if (sampleIndices.has(i)) examples.push({ text: row.text, cldrOnly, both, dataOnly })
  })

  console.log(`rows: ${lines.length}`)
  console.log(`cldr / data: ${cldrOnlyTotal}`)
  console.log(`cldr & data: ${bothTotal}`)
  console.log(`data / cldr: ${dataOnlyTotal}`)
  console.log()
  console.log(`${examples.length} random examples:`)
  for (const ex of examples) {
    console.log(`- ${JSON.stringify(ex.text)}`)
    console.log(`  cldr / data: ${ex.cldrOnly.join(" ") || "(none)"}`)
    console.log(`  cldr & data: ${ex.both.join(" ") || "(none)"}`)
    console.log(`  data / cldr: ${ex.dataOnly.join(" ") || "(none)"}`)
  }
  process.exit(0)
}
