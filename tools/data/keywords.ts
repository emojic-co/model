import { cac } from "cac"

import { CLDR_JSONL, EMOJILIB_JSONL, KEYWORDS_JSONL, LABELS_JSON, TERMS_JSONL } from "../../files.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"
import { normalize } from "./normalize.ts"

export type SourceRow = {
  text: string
  emojis: string
  styles: string[]
  bg?: [string, string]
  fg?: string
}
export type Src = "cldr" | "emojilib" | "both"
export type MergedRow = {
  text: string
  emojis: string[]
  styles: string[]
  bg?: [string, string]
  fg?: string
  src: Src
}

function byCodePoint(a: string, b: string): number {
  return (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0) || (a < b ? -1 : a > b ? 1 : 0)
}

export function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length
}

export function mergeKeywords(
  cldr: SourceRow[],
  emojilib: SourceRow[],
  rand: () => number = Math.random,
): MergedRow[] {
  type Palette = { bg: [string, string]; fg: string }
  type Acc = {
    emojis: Set<string>
    styles: Set<string>
    palettes: Palette[]
    srcs: Set<"cldr" | "emojilib">
  }
  const byText = new Map<string, Acc>()
  const add = (rows: SourceRow[], src: "cldr" | "emojilib") => {
    for (const r of rows) {
      const key = normalize(r.text)
      let a = byText.get(key)
      if (!a) byText.set(key, (a = { emojis: new Set(), styles: new Set(), palettes: [], srcs: new Set() }))
      for (const e of r.emojis.split(" ").filter(Boolean)) a.emojis.add(e)
      for (const s of r.styles) a.styles.add(s)
      if (r.bg && r.fg) a.palettes.push({ bg: r.bg, fg: r.fg })
      a.srcs.add(src)
    }
  }
  add(cldr, "cldr")
  add(emojilib, "emojilib")

  const out: MergedRow[] = []
  for (const [text, a] of byText) {
    const rec: MergedRow = {
      text,
      emojis: [...a.emojis].sort(byCodePoint),
      styles: [...a.styles],
      src: a.srcs.size === 2 ? "both" : [...a.srcs][0],
    }
    if (a.palettes.length) {
      const p = a.palettes[Math.floor(rand() * a.palettes.length)]
      rec.bg = p.bg
      rec.fg = p.fg
    }
    out.push(rec)
  }
  return out.sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))
}

export type KeywordsAndTerms = { keywords: MergedRow[]; terms: MergedRow[] }

export function splitKeywordsAndTerms(
  merged: MergedRow[],
  vocab: Set<string>,
): KeywordsAndTerms {
  const keywords: MergedRow[] = []
  const terms: MergedRow[] = []
  for (const r of merged) {
    const emojis = r.emojis.filter((e) => vocab.has(e))
    if (!emojis.length) continue
    if (r.text.length < 3) continue
    const rec = { ...r, emojis }
    ;(wordCount(r.text) === 1 ? keywords : terms).push(rec)
  }
  return { keywords, terms }
}

function toLine(r: MergedRow): string {
  const base: Record<string, unknown> = {
    text: r.text,
    emojis: r.emojis.join(" "),
    styles: r.styles,
  }
  if (r.bg && r.fg) {
    base.bg = r.bg
    base.fg = r.fg
  }
  base.src = r.src
  return JSON.stringify(base)
}

export async function buildKeywordsAndTerms(vocab: Set<string>): Promise<KeywordsAndTerms> {
  const [cldr, emojilib] = await Promise.all([
    readJsonl<SourceRow>(CLDR_JSONL),
    readJsonl<SourceRow>(EMOJILIB_JSONL),
  ])
  const merged = mergeKeywords(cldr, emojilib)
  return splitKeywordsAndTerms(merged, vocab)
}

export async function writeKeywordsAndTerms(vocab: Set<string>): Promise<KeywordsAndTerms> {
  const { keywords, terms } = await buildKeywordsAndTerms(vocab)
  await writeFileAtomic(KEYWORDS_JSONL, keywords.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TERMS_JSONL, terms.map(toLine).join("\n") + "\n")
  return { keywords, terms }
}

if (import.meta.main) {
  const cli = cac("keywords")
  cli.usage("[options]")
  cli.help()
  cli.parse(process.argv, { run: false })

  const { emojis } = JSON.parse(await Bun.file(LABELS_JSON).text()) as { emojis: string[] }
  const { keywords, terms } = await writeKeywordsAndTerms(new Set(emojis))

  const counts = (rows: MergedRow[]) => {
    const c = { cldr: 0, emojilib: 0, both: 0 }
    for (const r of rows) c[r.src]++
    return c
  }
  const kc = counts(keywords)
  const tc = counts(terms)
  console.log(
    `-> ${KEYWORDS_JSONL} : ${keywords.length} (cldr ${kc.cldr}, emojilib ${kc.emojilib}, both ${kc.both})`,
  )
  console.log(
    `-> ${TERMS_JSONL} : ${terms.length} (cldr ${tc.cldr}, emojilib ${tc.emojilib}, both ${tc.both})`,
  )
  process.exit(0)
}
