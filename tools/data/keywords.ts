import { cac } from "cac"

import { CLDR_JSONL, EMOJILIB_JSONL, KEYWORDS_JSONL } from "../../files.ts"
import { FALLBACK_PALETTES } from "./annotate.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"

type SourceRow = { text: string; emojis: string; styles: string[] }
type Src = "cldr" | "emojilib" | "both"
export type MergedRow = { text: string; emojis: string[]; styles: string[]; src: Src }

function byCodePoint(a: string, b: string): number {
  return (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
}

export function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length
}

export function randomPalette(
  rand: () => number = Math.random,
): { bg: [string, string]; fg: string } {
  const p = FALLBACK_PALETTES[Math.floor(rand() * FALLBACK_PALETTES.length)]
  return { bg: [p.bg[0], p.bg[1]], fg: p.fg }
}

export function mergeKeywords(cldr: SourceRow[], emojilib: SourceRow[]): MergedRow[] {
  type Acc = { emojis: Set<string>; styles: Set<string>; srcs: Set<"cldr" | "emojilib"> }
  const byText = new Map<string, Acc>()
  const add = (rows: SourceRow[], src: "cldr" | "emojilib") => {
    for (const r of rows) {
      let a = byText.get(r.text)
      if (!a) byText.set(r.text, (a = { emojis: new Set(), styles: new Set(), srcs: new Set() }))
      for (const e of r.emojis.split(" ").filter(Boolean)) a.emojis.add(e)
      for (const s of r.styles) a.styles.add(s)
      a.srcs.add(src)
    }
  }
  add(cldr, "cldr")
  add(emojilib, "emojilib")

  const out: MergedRow[] = []
  for (const [text, a] of byText) {
    out.push({
      text,
      emojis: [...a.emojis].sort(byCodePoint),
      styles: [...a.styles],
      src: a.srcs.size === 2 ? "both" : [...a.srcs][0],
    })
  }
  return out.sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))
}

if (import.meta.main) {
  const cli = cac("keywords")
  cli.usage("[options]")
  cli.help()
  cli.parse(process.argv, { run: false })

  const [cldr, emojilib] = await Promise.all([
    readJsonl<SourceRow>(CLDR_JSONL),
    readJsonl<SourceRow>(EMOJILIB_JSONL),
  ])
  const merged = mergeKeywords(cldr, emojilib)
  const lines = merged.map((r) => {
    const { bg, fg } = randomPalette()
    return JSON.stringify({
      text: r.text,
      emojis: r.emojis.join(" "),
      styles: r.styles,
      bg,
      fg,
      src: r.src,
      word_count: wordCount(r.text),
    })
  })
  await writeFileAtomic(KEYWORDS_JSONL, lines.join("\n") + "\n")

  const counts = { cldr: 0, emojilib: 0, both: 0 }
  for (const r of merged) counts[r.src]++
  console.log(
    `${cldr.length} cldr + ${emojilib.length} emojilib -> ${merged.length} keywords `
    + `(cldr ${counts.cldr}, emojilib ${counts.emojilib}, both ${counts.both}) `
    + `-> ${KEYWORDS_JSONL}`,
  )
  process.exit(0)
}
