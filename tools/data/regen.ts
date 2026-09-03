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
}

type Acc = {
  text: string
  emojis: Set<string>
  styles: Set<string>
  palettes: Palette[]
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
      a = { text, emojis: new Set(), styles: new Set(), palettes: [] }
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
    out.push(rec)
  }
  return out
}

export function shuffleSeeded<T>(rows: T[], seed: number): T[] {
  let s = seed >>> 0
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  const out = [...rows]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

export function emojiLeaderboard(
  records: { emojis: string }[],
  n: number,
): string[] {
  const counts = new Map<string, number>()
  for (const rec of records) {
    for (const e of new Set(splitEmojis(rec.emojis))) {
      counts.set(e, (counts.get(e) ?? 0) + 1)
    }
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k]) => k)
}

import { readJsonl, writeFileAtomic } from "./io.ts"
import { STYLES, TOP_EMOJIS } from "./config"

const DATA = "./data.jsonl"
const TRAIN = "./train.jsonl"
const EVAL = "./eval.jsonl"
const LABELS = "./labels.json"

function argInt(flag: string): number | undefined {
  const i = process.argv.indexOf(flag)
  if (i < 0 || i + 1 >= process.argv.length) return undefined
  const n = Number(process.argv[i + 1])
  return Number.isFinite(n) ? n : undefined
}

function toLine(r: Row): string {
  return JSON.stringify(
    r.bg && r.fg
      ? { text: r.text, emojis: r.emojis, styles: r.styles, bg: r.bg, fg: r.fg }
      : { text: r.text, emojis: r.emojis, styles: r.styles },
  )
}

if (import.meta.main) {
  const seed = argInt("--seed") ?? 42
  const n = argInt("--n") ?? 1500

  const raw = await readJsonl<unknown>(DATA)
  const records = collapse(raw)
  const merged = records.length
  const dupKeys = raw.length - merged

  const shuffled = shuffleSeeded(records, seed)
  const held = shuffled.slice(0, n)
  const rest = shuffled.slice(n)

  await writeFileAtomic(EVAL, held.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TRAIN, rest.map(toLine).join("\n") + "\n")

  const labels = {
    styles: [...STYLES],
    emojis: emojiLeaderboard(records, TOP_EMOJIS),
  }
  await writeFileAtomic(LABELS, JSON.stringify(labels, null, 2) + "\n")

  console.log("\n--- regen ---")
  console.log(`master lines read    : ${raw.length}`)
  console.log(`distinct keys        : ${merged}`)
  console.log(`collapsed away       : ${dupKeys}`)
  console.log(`-> ${EVAL}      : ${held.length}`)
  console.log(`-> ${TRAIN}     : ${rest.length}`)
  console.log(
    `-> ${LABELS}   : ${labels.styles.length} styles, ${labels.emojis.length} emojis`,
  )
  process.exit(0)
}
