import { existsSync } from "node:fs"

import { splitEmojis } from "./emoji.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"
import { normalize } from "./normalize.ts"
import { stableHash } from "./pool.ts"
import { STYLE_SET } from "./styles.ts"

const FILES = [
  { path: "./eval.jsonl", tier: 0 },
  { path: "./train.jsonl", tier: 1 },
  { path: "./data.jsonl", tier: 2 },
] as const

type RawRow = Record<string, unknown>
type Color = { bg: string[]; fg: string }
type Acc = {
  tier: number
  text: string
  emojis: Set<string>
  styles: Set<string>
  colors: Color[][]
  files: Set<number>
}

function rowEmojis(row: RawRow): string[] {
  const out: string[] = []
  for (const field of [row.emoji, row.emojis]) {
    if (typeof field === "string") out.push(...splitEmojis(field))
    else if (Array.isArray(field)) out.push(...splitEmojis(field.join(" ")))
  }
  return out
}

function rowStyles(row: RawRow): string[] {
  const out: string[] = []
  for (const field of [row.style, row.styles, row.feeling, row.feelings]) {
    const vals =
      typeof field === "string"
        ? [field]
        : Array.isArray(field)
          ? field.filter((v): v is string => typeof v === "string")
          : []
    for (const v of vals) if (STYLE_SET.has(v)) out.push(v)
  }
  return out
}

function rowColor(row: RawRow): Color | undefined {
  if (Array.isArray(row.bg) && row.bg.length && typeof row.fg === "string") {
    return { bg: row.bg as string[], fg: row.fg }
  }
  return undefined
}

function pickColor(key: string, a: Acc): Color | undefined {
  for (let t = a.tier; t < FILES.length; t++) {
    const cs = a.colors[t]
    if (cs.length) {
      const r = ((stableHash(key) * 1664525 + 1013904223) >>> 0) / 2 ** 32
      return cs[Math.floor(r * cs.length)]
    }
  }
  return undefined
}

if (import.meta.main) {
  const acc = new Map<string, Acc>()
  const readCount: Record<string, number> = {}
  let dropped = 0

  for (const { path, tier } of FILES) {
    if (!existsSync(path)) {
      readCount[path] = 0
      continue
    }
    const rows = await readJsonl<RawRow>(path)
    readCount[path] = rows.length
    for (const row of rows) {
      const text = typeof row.text === "string" ? row.text : ""
      const key = normalize(text)
      if (!key) {
        dropped++
        continue
      }
      let a = acc.get(key)
      if (!a) {
        a = {
          tier,
          text,
          emojis: new Set(),
          styles: new Set(),
          colors: [[], [], []],
          files: new Set(),
        }
        acc.set(key, a)
      }
      a.files.add(tier)
      for (const e of rowEmojis(row)) a.emojis.add(e)
      for (const s of rowStyles(row)) a.styles.add(s)
      const c = rowColor(row)
      if (c) a.colors[tier].push(c)
    }
  }

  const out: string[][] = FILES.map(() => [])
  const emojiVocab = new Set<string>()
  const styleVocab = new Set<string>()
  let noColor = 0
  let noEmoji = 0
  let noStyle = 0
  let multiSource = 0
  for (const [key, a] of acc) {
    const emojis = [...a.emojis]
    const styles = [...a.styles]
    const rec: Record<string, unknown> = { text: a.text, emojis: emojis.join(" "), styles }
    const c = pickColor(key, a)
    if (c) {
      rec.bg = c.bg
      rec.fg = c.fg
    } else {
      noColor++
    }
    if (!emojis.length) noEmoji++
    if (!styles.length) noStyle++
    if (a.files.size > 1) multiSource++
    for (const e of emojis) emojiVocab.add(e)
    for (const s of styles) styleVocab.add(s)
    out[a.tier].push(JSON.stringify(rec))
  }

  for (const { path, tier } of FILES) {
    await writeFileAtomic(path, out[tier].length ? out[tier].join("\n") + "\n" : "", true)
  }

  const totalRead = FILES.reduce((n, { path }) => n + readCount[path], 0)
  console.log("\n===== merge summary =====")
  for (const { path } of FILES)
    console.log(`  read  ${path.padEnd(14)} : ${readCount[path]}`)
  console.log(`  read  total          : ${totalRead}`)
  console.log(`  dropped (empty norm) : ${dropped}`)
  console.log(`  unique normalized    : ${acc.size}`)
  console.log(`  merged from 2+ files : ${multiSource}`)
  console.log("  ---")
  for (const { path, tier } of FILES)
    console.log(`  wrote ${path.padEnd(14)} : ${out[tier].length}`)
  const written = out.reduce((n, b) => n + b.length, 0)
  console.log(`  wrote total          : ${written}  (removed ${totalRead - dropped - written} dup rows)`)
  console.log("  ---")
  console.log(`  distinct emojis      : ${emojiVocab.size}`)
  console.log(`  distinct styles      : ${styleVocab.size}`)
  console.log(`  records w/o emoji    : ${noEmoji}`)
  console.log(`  records w/o style    : ${noStyle}`)
  console.log(`  records w/o color    : ${noColor}`)
  process.exit(0)
}
