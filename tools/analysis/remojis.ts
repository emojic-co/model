import { DATA_JSONL } from "../../files.ts"
import { splitEmojis } from "../data/emoji.ts"
import { readJsonl } from "../data/io.ts"

const TOP_N = 25

export type RemojiRow = {
  emojis?: unknown
  remojis?: unknown
  meta?: unknown
}

export type RemojiStats = {
  records: number
  withAdditions: number
  zeroAdditions: number
  totalAdded: number
  meanAddedPerRecord: number
  meanAddedPerExpanded: number
  distinctAdded: number
  addedCountHistogram: { added: number; records: number }[]
  origCountHistogram: { orig: number; records: number }[]
  meanOrigCount: number
  meanFinalCount: number
  expandedFromEmpty: number
  expandedExisting: number
  topAdded: { emoji: string; count: number; pct: number }[]
  byDate: { date: string; records: number; totalAdded: number; meanAdded: number }[]
}

function toStringArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []
}

export function computeRemojiStats(rows: RemojiRow[]): RemojiStats {
  const withField = rows.filter((r) => Array.isArray(r.remojis))
  const records = withField.length

  const addedCounts = new Map<number, number>()
  const origCounts = new Map<number, number>()
  const addedFreq = new Map<string, number>()
  const dateAgg = new Map<string, { records: number; totalAdded: number }>()

  let totalAdded = 0
  let withAdditions = 0
  let origSum = 0
  let finalSum = 0
  let expandedFromEmpty = 0
  let expandedExisting = 0

  for (const row of withField) {
    const added = toStringArray(row.remojis)
    const final = splitEmojis(typeof row.emojis === "string" ? row.emojis : "")
    const k = added.length
    const orig = Math.max(0, final.length - k)

    totalAdded += k
    origSum += orig
    finalSum += final.length
    addedCounts.set(k, (addedCounts.get(k) ?? 0) + 1)
    origCounts.set(orig, (origCounts.get(orig) ?? 0) + 1)
    if (k > 0) {
      withAdditions++
      if (orig === 0) expandedFromEmpty++
      else expandedExisting++
    }
    for (const e of added) addedFreq.set(e, (addedFreq.get(e) ?? 0) + 1)

    const meta = (row.meta ?? {}) as Record<string, unknown>
    const date = typeof meta.date === "string" ? meta.date : "(none)"
    const agg = dateAgg.get(date) ?? { records: 0, totalAdded: 0 }
    agg.records++
    agg.totalAdded += k
    dateAgg.set(date, agg)
  }

  const denom = records || 1
  return {
    records,
    withAdditions,
    zeroAdditions: records - withAdditions,
    totalAdded,
    meanAddedPerRecord: totalAdded / denom,
    meanAddedPerExpanded: totalAdded / (withAdditions || 1),
    distinctAdded: addedFreq.size,
    addedCountHistogram: [...addedCounts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([added, r]) => ({ added, records: r })),
    origCountHistogram: [...origCounts.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([orig, r]) => ({ orig, records: r })),
    meanOrigCount: origSum / denom,
    meanFinalCount: finalSum / denom,
    expandedFromEmpty,
    expandedExisting,
    topAdded: [...addedFreq.entries()]
      .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
      .slice(0, TOP_N)
      .map(([emoji, count]) => ({ emoji, count, pct: (100 * count) / (totalAdded || 1) })),
    byDate: [...dateAgg.entries()]
      .sort((a, b) => (a[0] < b[0] ? -1 : 1))
      .map(([date, agg]) => ({
        date,
        records: agg.records,
        totalAdded: agg.totalAdded,
        meanAdded: agg.totalAdded / (agg.records || 1),
      })),
  }
}

function bar(n: number, max: number, width = 40): string {
  return "#".repeat(max > 0 ? Math.round((n / max) * width) : 0)
}

function printStats(s: RemojiStats): void {
  const pct = (n: number, d: number) => (d ? ((100 * n) / d).toFixed(1) : "0.0")
  console.log(`records with remojis field : ${s.records}`)
  console.log(
    `  with >=1 addition        : ${s.withAdditions} (${pct(s.withAdditions, s.records)}%)`,
  )
  console.log(
    `  no addition (remojis [])  : ${s.zeroAdditions} (${pct(s.zeroAdditions, s.records)}%)`,
  )
  console.log(`new emojis added (total)   : ${s.totalAdded}`)
  console.log(`  per record (all)         : ${s.meanAddedPerRecord.toFixed(3)}`)
  console.log(`  per expanded record      : ${s.meanAddedPerExpanded.toFixed(3)}`)
  console.log(`distinct emojis added      : ${s.distinctAdded}`)
  console.log(
    `mean emoji count           : ${s.meanOrigCount.toFixed(2)} -> ${s.meanFinalCount.toFixed(2)}`,
  )
  console.log(
    `expanded from empty / existing : ${s.expandedFromEmpty} / ${s.expandedExisting}`,
  )

  console.log("\nadditions per record:")
  const acMax = Math.max(...s.addedCountHistogram.map((b) => b.records), 1)
  for (const b of s.addedCountHistogram) {
    console.log(
      `  ${String(b.added).padStart(2)} | ${String(b.records).padStart(7)} `
      + `${bar(b.records, acMax)}`,
    )
  }

  console.log("\noriginal emoji count (expanded rows included):")
  const ocMax = Math.max(...s.origCountHistogram.map((b) => b.records), 1)
  for (const b of s.origCountHistogram) {
    console.log(
      `  ${String(b.orig).padStart(2)} | ${String(b.records).padStart(7)} `
      + `${bar(b.records, ocMax)}`,
    )
  }

  if (s.byDate.length > 1) {
    console.log("\nby meta.date:")
    for (const d of s.byDate) {
      console.log(
        `  ${d.date.padEnd(12)} ${String(d.records).padStart(7)} rows  `
        + `${d.totalAdded.toString().padStart(6)} added  `
        + `${d.meanAdded.toFixed(3)}/row`,
      )
    }
  }

  console.log(`\ntop ${TOP_N} added emojis:`)
  for (const e of s.topAdded) {
    console.log(
      `  ${e.emoji}  ${String(e.count).padStart(6)}  ${e.pct.toFixed(1).padStart(5)}%`,
    )
  }
}

if (import.meta.main) {
  const rows = await readJsonl<RemojiRow>(DATA_JSONL)
  const stats = computeRemojiStats(rows)
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify(stats, null, 2))
  } else {
    console.log(`source: ${DATA_JSONL} (${rows.length} rows)\n`)
    printStats(stats)
  }
}
