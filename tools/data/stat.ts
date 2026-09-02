import { existsSync } from "node:fs"
import { mkdir, writeFile } from "node:fs/promises"

import { STYLES, TOP_EMOJIS } from "./config"
import { coarseEmojiGroup, isFaceEmoji, splitEmojis } from "./emoji.ts"
import { readJsonl } from "./io.ts"

const FILES = ["./train.jsonl", "./eval.jsonl"]
const REPORT_DIR = "report/data-stat"

const MIN_LEN = 4
const MAX_LEN = 48
const BUCKET = 5

type Row = { text: string; emojis?: string; styles?: string[] }

function emojiList(r: Row): string[] {
  return typeof r.emojis === "string" ? splitEmojis(r.emojis) : []
}

function styleList(r: Row): string[] {
  return Array.isArray(r.styles) ? r.styles : []
}

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

function countHistogram(values: number[]): string[] {
  if (!values.length) return ["_(no rows)_"]
  const by = counts(values)
  const hi = Math.max(...by.keys())
  const peak = Math.max(...by.values())
  const out = ["| count | rows | |", "| ---: | ---: | :-- |"]
  for (let b = 0; b <= hi; b++) {
    const n = by.get(b) ?? 0
    const bar = n ? "#".repeat(Math.round((40 * n) / peak)) : ""
    out.push(`| ${b} | ${n} | ${bar} |`)
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

function coverage(rows: Row[]): string {
  if (!rows.length) return "_(no rows)_"
  const topEmojis = new Set(
    ranked(counts(rows.flatMap((r) => [...new Set(emojiList(r))])))
      .slice(0, TOP_EMOJIS)
      .map(([e]) => e),
  )
  const pct = (n: number) => `${((100 * n) / rows.length).toFixed(1)}%`
  const anyIn = rows.filter((r) =>
    emojiList(r).some((e) => topEmojis.has(e)),
  ).length
  const allIn = rows.filter((r) => {
    const l = emojiList(r)
    return l.length > 0 && l.every((e) => topEmojis.has(e))
  }).length
  const styleOk = rows.filter((r) => {
    const l = styleList(r)
    return l.length > 0 && l.every((s) => (STYLES as readonly string[]).includes(s))
  }).length
  return `≥1 emoji in top-${TOP_EMOJIS} ${pct(anyIn)} · all emojis in top ${pct(allIn)} · styles valid ${pct(styleOk)}`
}

function section(path: string, rows: Row[]): string[] {
  const out = [`## \`${path}\``, "", `**${rows.length} rows**`, ""]
  if (!rows.length) return out

  const styleMentions = rows.flatMap(styleList)
  const sc = ranked(counts(styleMentions))
  out.push("### Style distribution (multi-label)", "")
  out.push("|  | style | rows | share |", "| --- | --- | ---: | ---: |")
  sc.forEach(([s, n], i) => {
    out.push(`| ${i + 1} | ${s} | ${n} | ${((100 * n) / rows.length).toFixed(1)}% |`)
  })
  out.push("")
  out.push("Styles per row:", "")
  out.push(...countHistogram(rows.map((r) => styleList(r).length)), "")

  const lens = rows.map((r) => [...r.text].length)
  const outOfRange = lens.filter((l) => l < MIN_LEN || l > MAX_LEN).length
  out.push("### Text length (code points)", "")
  out.push(summary(lens), "")
  out.push(
    `${outOfRange} rows outside ${MIN_LEN}–${MAX_LEN} (${((100 * outOfRange) / rows.length).toFixed(1)}%)`,
    "",
  )
  out.push(...histogram(lens), "")

  const allMentions = rows.flatMap(emojiList)
  const ec = ranked(counts(allMentions))
  const singletons = ec.filter(([, n]) => n === 1).length
  const faceMentions = allMentions.filter(isFaceEmoji).length
  const rowsWithFace = rows.filter((r) => emojiList(r).some(isFaceEmoji)).length
  out.push("### Emoji distribution (multi-label)", "")
  out.push(
    `${ec.length} distinct emojis · ${singletons} appear once · ${allMentions.length} total mentions`,
    "",
  )
  out.push("Emojis per row:", "")
  out.push(...countHistogram(rows.map((r) => emojiList(r).length)), "")
  const noEmoji = rows.filter((r) => emojiList(r).length === 0).length
  out.push(
    `**No emoji: ${((100 * noEmoji) / rows.length).toFixed(1)}% of rows**`,
    "",
  )
  out.push(
    `**Face emojis: ${((100 * faceMentions) / (allMentions.length || 1)).toFixed(1)}% of mentions · ${((100 * rowsWithFace) / rows.length).toFixed(1)}% of rows**`,
    "",
  )
  const gc = ranked(counts(allMentions.map(coarseEmojiGroup)))
  out.push("Unicode group coverage (by mention):", "")
  out.push("| group | mentions | share |", "| --- | ---: | ---: |")
  gc.forEach(([g, n]) => {
    out.push(`| ${g} | ${n} | ${((100 * n) / (allMentions.length || 1)).toFixed(1)}% |`)
  })
  out.push("")
  out.push("| # | emoji | rows | share |", "| ---: | --- | ---: | ---: |")
  const docFreq = ranked(counts(rows.flatMap((r) => [...new Set(emojiList(r))])))
  docFreq.slice(0, 30).forEach(([e, n], i) => {
    out.push(`| ${i + 1} | ${e} | ${n} | ${((100 * n) / rows.length).toFixed(1)}% |`)
  })
  out.push("")
  return out
}

if (import.meta.main) {
  const { header, file } = stamp()
  const doc = [`# data stats — ${header}`, ""]

  const parsed: { path: string; rows: Row[] | null }[] = []
  for (const path of FILES) {
    if (!existsSync(path)) {
      parsed.push({ path, rows: null })
      continue
    }
    parsed.push({ path, rows: await readJsonl<Row>(path) })
  }

  doc.push(
    "## Top-label coverage",
    "",
    `${STYLES.length} styles · top ${TOP_EMOJIS} emojis (from \`config.ts\`)`,
    "",
  )
  for (const { path, rows } of parsed) {
    doc.push(`- \`${path}\` — ${rows ? coverage(rows) : "_(missing)_"}`)
  }
  doc.push("")

  for (const { path, rows } of parsed) {
    if (!rows) {
      doc.push(`## \`${path}\``, "", "_(missing)_", "")
      continue
    }
    doc.push(...section(path, rows))
  }

  await mkdir(REPORT_DIR, { recursive: true })
  const dest = `${REPORT_DIR}/${file}.md`
  await writeFile(dest, doc.join("\n"))
  console.log(dest)
  process.exit(0)
}
