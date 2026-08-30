import { existsSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"

const FILES = ["./train.jsonl", "./eval.jsonl"]
const REPORT_DIR = "./report/data-stat"

const MIN_LEN = 4
const MAX_LEN = 48
const BUCKET = 5

type Row = { text: string; feeling: string; emoji: string }

function stamp(): { header: string; file: string } {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return {
    header: `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`,
    file: `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}:${p(d.getMinutes())}`,
  }
}

function counts<T>(items: T[]): Map<T, number> {
  const m = new Map<T, number>()
  for (const it of items) m.set(it, (m.get(it) ?? 0) + 1)
  return m
}

function ranked<T>(m: Map<T, number>): [T, number][] {
  return [...m.entries()].sort((a, b) => b[1] - a[1])
}

function histogram(values: number[]): string[] {
  if (!values.length) return ["_(no rows)_"]
  const by = counts(values.map((v) => Math.floor(v / BUCKET)))
  const hi = Math.max(...by.keys())
  const peak = Math.max(...by.values())
  const out = ["| range | count | |", "| --- | ---: | :-- |"]
  for (let b = 0; b <= hi; b++) {
    const n = by.get(b) ?? 0
    const bar = n ? "#".repeat(Math.round((40 * n) / peak)) : ""
    out.push(`| ${b * BUCKET}–${b * BUCKET + BUCKET - 1} | ${n} | ${bar} |`)
  }
  return out
}

function summary(values: number[]): string {
  const s = [...values].sort((a, b) => a - b)
  const n = s.length
  if (!n) return "_(no rows)_"
  const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
  const mean = s.reduce((a, b) => a + b, 0) / n
  return `min ${s[0]} · median ${median} · mean ${mean.toFixed(1)} · max ${s[n - 1]}`
}

function section(path: string, rows: Row[]): string[] {
  const out = [`## \`${path}\``, "", `**${rows.length} rows**`, ""]
  if (!rows.length) return out

  const fc = ranked(counts(rows.map((r) => r.feeling)))
  out.push("### Feeling distribution", "")
  out.push("| feeling | count | share |", "| --- | ---: | ---: |")
  for (const [f, n] of fc) {
    out.push(`| ${f} | ${n} | ${((100 * n) / rows.length).toFixed(1)}% |`)
  }
  out.push("")

  const lens = rows.map((r) => [...r.text].length)
  const outOfRange = lens.filter((l) => l < MIN_LEN || l > MAX_LEN).length
  out.push("### Text length (code points)", "")
  out.push(summary(lens), "")
  out.push(
    `${outOfRange} rows outside ${MIN_LEN}–${MAX_LEN} (${((100 * outOfRange) / rows.length).toFixed(1)}%)`,
    "",
  )
  out.push(...histogram(lens), "")

  const ec = ranked(counts(rows.map((r) => r.emoji)))
  const singletons = ec.filter(([, n]) => n === 1).length
  out.push("### Emoji distribution", "")
  out.push(
    `${ec.length} distinct emojis · ${singletons} appear once`,
    "",
    "| # | emoji | count | share |",
    "| ---: | --- | ---: | ---: |",
  )
  ec.slice(0, 30).forEach(([e, n], i) => {
    out.push(`| ${i + 1} | ${e} | ${n} | ${((100 * n) / rows.length).toFixed(1)}% |`)
  })
  out.push("")
  return out
}

if (import.meta.main) {
  const { header, file } = stamp()
  const doc = [`# data stats — ${header}`, ""]

  for (const path of FILES) {
    if (!existsSync(path)) {
      doc.push(`## \`${path}\``, "", "_(missing)_", "")
      continue
    }
    const rows = (await readFile(path, "utf8"))
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => JSON.parse(l) as Row)
    doc.push(...section(path, rows))
  }

  await mkdir(REPORT_DIR, { recursive: true })
  const dest = `${REPORT_DIR}/${file}.md`
  await writeFile(dest, doc.join("\n"))
  console.log(`wrote ${dest}`)
  process.exit(0)
}
