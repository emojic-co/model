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
