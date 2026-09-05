import { cac } from "cac"

import { splitEmojis } from "./emoji.ts"
import { normalize } from "./normalize.ts"
import { stableHash } from "./pool.ts"
import { STYLE_SET } from "./styles.ts"

export type Palette = { bg: string[]; fg: string }
export type Row = {
  text: string
  emojis: string
  styles: string[]
  bg?: string[]
  fg?: string
  extra?: Record<string, unknown>
}

const BASE_FIELDS = new Set(["text", "emojis", "styles", "bg", "fg"])

type Acc = {
  text: string
  emojis: Set<string>
  styles: Set<string>
  palettes: Palette[]
  extra: Record<string, unknown>
}

function rowPalette(row: Record<string, unknown>): Palette | undefined {
  const { bg, fg } = row
  if (Array.isArray(bg) && bg.length >= 2 && typeof fg === "string") {
    return { bg: (bg as string[]).slice(0, 2), fg }
  }
  return undefined
}

export function pickPalette(
  key: string,
  palettes: Palette[],
): Palette | undefined {
  if (!palettes.length) return undefined
  const r = ((stableHash(key) * 1664525 + 1013904223) >>> 0) / 2 ** 32
  return palettes[Math.floor(r * palettes.length)]
}

export function collapse(rows: unknown[]): Row[] {
  const acc = new Map<string, Acc>()
  for (const raw of rows) {
    const row = (raw ?? {}) as Record<string, unknown>
    const text = typeof row.text === "string" ? row.text : ""
    const key = normalize(text)
    if (!key) continue
    let a = acc.get(key)
    if (!a) {
      a = {
        text,
        emojis: new Set(),
        styles: new Set(),
        palettes: [],
        extra: {},
      }
      acc.set(key, a)
    }
    if (typeof row.emojis === "string") {
      for (const e of splitEmojis(row.emojis)) a.emojis.add(e)
    }
    if (Array.isArray(row.styles)) {
      for (const s of row.styles) {
        if (typeof s === "string" && STYLE_SET.has(s)) a.styles.add(s)
      }
    }
    const p = rowPalette(row)
    if (p) a.palettes.push(p)
    for (const [k, v] of Object.entries(row)) {
      if (BASE_FIELDS.has(k) || v === undefined || k in a.extra) continue
      a.extra[k] = v
    }
  }

  const out: Row[] = []
  for (const [key, a] of acc) {
    const rec: Row = {
      text: a.text,
      emojis: [...a.emojis].join(" "),
      styles: [...a.styles],
    }
    const p = pickPalette(key, a.palettes)
    if (p) {
      rec.bg = p.bg
      rec.fg = p.fg
    }
    if (Object.keys(a.extra).length) rec.extra = a.extra
    out.push(rec)
  }
  return out
}

export function shuffle<T>(rows: T[]): T[] {
  const out = [...rows]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function greedyCap(
  records: Row[],
  maxCount: number,
): { kept: Row[]; counts: Map<string, number>; dropped: number } {
  const counts = new Map<string, number>()
  const kept: Row[] = []
  for (const r of records) {
    const es = [...new Set(splitEmojis(r.emojis))]
    if (es.some((e) => (counts.get(e) ?? 0) >= maxCount)) continue
    for (const e of es) counts.set(e, (counts.get(e) ?? 0) + 1)
    kept.push(r)
  }
  return { kept, counts, dropped: records.length - kept.length }
}

export function emojiVocab(counts: Map<string, number>, minCount: number): string[] {
  return [...counts.entries()]
    .filter(([, c]) => c >= minCount)
    .sort((a, b) => b[1] - a[1])
    .map(([e]) => e)
}

import {
  DATA_JSONL as DATA,
  EVAL_JSONL as EVAL,
  LABELS_JSON as LABELS,
  TRAIN_JSONL as TRAIN,
} from "../../files.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"
import { STYLES } from "./config"

export function toLine(r: Row): string {
  const base =
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles }
  return JSON.stringify(r.extra ? { ...base, ...r.extra } : base)
}

const cli = cac("regen")
cli.usage("[options]")
cli
  .option("--min-count <n>", "min kept-records for an emoji to enter labels.json (default 100)")
  .option("--max-count <n>", "cap on kept-records per emoji (default 500)")
  .option("--n <n>", "eval.jsonl row count (default 1500)")
cli.help()

if (import.meta.main) {
  const { options } = cli.parse(process.argv, { run: false })
  if (options.help) process.exit(0)
  const minCount = Number(options.minCount ?? 100)
  const maxCount = Number(options.maxCount ?? 500)
  const n = Number(options.n ?? 1500)

  const raw = await readJsonl<unknown>(DATA)
  const records = collapse(raw)
  const merged = records.length
  const dupKeys = raw.length - merged

  const { kept, counts, dropped } = greedyCap(shuffle(records), maxCount)
  const emojis = emojiVocab(counts, minCount)
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1])
  const belowMin = ranked.length - emojis.length

  const split = shuffle(kept)
  const held = split.slice(0, n)
  const rest = split.slice(n)

  await writeFileAtomic(EVAL, held.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TRAIN, rest.map(toLine).join("\n") + "\n")

  const labels = {
    styles: [...STYLES],
    emojis,
  }
  await writeFileAtomic(LABELS, JSON.stringify(labels, null, 2) + "\n")

  const keptRanked = ranked.filter(([, c]) => c >= minCount)
  const fmt = (es: [string, number][]) => es.map(([e, c]) => `${e} ${c}`).join(", ")

  console.log("\n--- regen ---")
  console.log(`master lines read     : ${raw.length}`)
  console.log(`distinct texts        : ${merged} (collapsed away ${dupKeys})`)
  console.log(`min-count / max-count : ${minCount} / ${maxCount}`)
  console.log(`greedy kept           : ${kept.length} rows (dropped ${dropped} over max-count)`)
  console.log(`emoji vocab (>= ${minCount})  : ${emojis.length} (below min-count: ${belowMin})`)
  console.log(`  most frequent       : ${fmt(keptRanked.slice(0, 5))}`)
  console.log(`  least frequent kept : ${fmt(keptRanked.slice(-5))}`)
  console.log(`-> ${EVAL}       : ${held.length}`)
  console.log(`-> ${TRAIN}      : ${rest.length}`)
  console.log(
    `-> ${LABELS}    : ${labels.styles.length} styles, ${labels.emojis.length} emojis`,
  )
  process.exit(0)
}
