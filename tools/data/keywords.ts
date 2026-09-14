import { cac } from "cac"

import { CLDR_JSONL, EMOJILIB_JSONL, FLAGS_JSONL, KEYWORDS_JSONL, LABELS_JSON, TERMS_JSONL } from "../../files.ts"
import { isFlagEmoji, splitEmojis } from "./emoji.ts"
import { readJsonl, writeFileAtomic } from "./io.ts"
import { normalize } from "./normalize.ts"
import { loadWaKeywords } from "./wa-keywords.ts"

export type SourceRow = {
  text: string
  emojis: string
  styles: string[]
  bg?: [string, string]
  fg?: string
}
export type Src = string
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
  sources: Record<string, SourceRow[]>,
  rand: () => number = Math.random,
): MergedRow[] {
  type Palette = { bg: [string, string]; fg: string }
  type Acc = {
    emojis: Set<string>
    styles: Set<string>
    palettes: Palette[]
    srcs: Set<string>
  }
  const byText = new Map<string, Acc>()
  const add = (rows: SourceRow[], src: string) => {
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
  for (const [name, rows] of Object.entries(sources)) add(rows, name)

  const out: MergedRow[] = []
  for (const [text, a] of byText) {
    const rec: MergedRow = {
      text,
      emojis: [...a.emojis].sort(byCodePoint),
      styles: [...a.styles],
      src: [...a.srcs].sort().join("+"),
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
    if (emojis.every(isFlagEmoji)) continue
    const rec = { ...r, emojis }
    ;(wordCount(r.text) === 1 ? keywords : terms).push(rec)
  }
  return { keywords, terms }
}

export function buildFlags(cldr: SourceRow[], vocab: Set<string>): MergedRow[] {
  const seen = new Set<string>()
  const out: MergedRow[] = []
  for (const r of cldr) {
    const emojis = splitEmojis(r.emojis)
    if (emojis.length !== 1 || !isFlagEmoji(emojis[0])) continue
    if (!vocab.has(emojis[0])) continue
    if (!r.bg || !r.fg) continue
    const key = normalize(r.text)
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push({
      text: r.text,
      emojis,
      styles: r.styles,
      bg: r.bg,
      fg: r.fg,
      src: "cldr",
    })
  }
  return out.sort((a, b) => (a.text < b.text ? -1 : a.text > b.text ? 1 : 0))
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

export type KeywordsTermsAndFlags = KeywordsAndTerms & { flags: MergedRow[] }

export async function buildKeywordsAndTerms(vocab: Set<string>): Promise<KeywordsTermsAndFlags> {
  const [cldr, emojilib, wa] = await Promise.all([
    readJsonl<SourceRow>(CLDR_JSONL),
    readJsonl<SourceRow>(EMOJILIB_JSONL),
    loadWaKeywords(),
  ])
  const merged = mergeKeywords({ cldr, emojilib, wa })
  const { keywords, terms } = splitKeywordsAndTerms(merged, vocab)
  const flags = buildFlags(cldr, vocab)
  return { keywords, terms, flags }
}

export async function writeKeywordsAndTerms(vocab: Set<string>): Promise<KeywordsTermsAndFlags> {
  const { keywords, terms, flags } = await buildKeywordsAndTerms(vocab)
  await writeFileAtomic(KEYWORDS_JSONL, keywords.map(toLine).join("\n") + "\n")
  await writeFileAtomic(TERMS_JSONL, terms.map(toLine).join("\n") + "\n")
  await writeFileAtomic(FLAGS_JSONL, flags.map(toLine).join("\n") + "\n")
  return { keywords, terms, flags }
}

if (import.meta.main) {
  const cli = cac("keywords")
  cli.usage("[options]")
  cli.help()
  cli.parse(process.argv, { run: false })

  const { emojis } = JSON.parse(await Bun.file(LABELS_JSON).text()) as { emojis: string[] }
  const { keywords, terms, flags } = await writeKeywordsAndTerms(new Set(emojis))

  const counts = (rows: MergedRow[]) => {
    const c = new Map<string, number>()
    for (const r of rows) c.set(r.src, (c.get(r.src) ?? 0) + 1)
    return c
  }
  const fmtCounts = (c: Map<string, number>) =>
    [...c.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([k, n]) => `${k} ${n}`).join(", ")
  console.log(`-> ${KEYWORDS_JSONL} : ${keywords.length} (${fmtCounts(counts(keywords))})`)
  console.log(`-> ${TERMS_JSONL} : ${terms.length} (${fmtCounts(counts(terms))})`)
  console.log(`-> ${FLAGS_JSONL} : ${flags.length}`)
  process.exit(0)
}
