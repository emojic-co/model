import { Index } from "flexsearch"

import { EVAL_JSONL, LABELS_JSON } from "../../files.ts"
import { readJsonl } from "../data/io.ts"
import { loadCldrAnnotations } from "../data/cldr.ts"

const VARIATION_SELECTORS = /[︎️]/g
const SEARCH_LIMIT = 50
const KS = Array.from({ length: 10 }, (_, i) => i + 1)
const TOKENIZERS = ["strict", "forward"] as const

export function stripVS(s: string): string {
  return s.replace(VARIATION_SELECTORS, "")
}

export function rowTargets(emojis: string, vocab: Set<string>): string[] {
  const out: string[] = []
  for (const raw of emojis.split(/\s+/)) {
    const e = stripVS(raw)
    if (e && vocab.has(e) && !out.includes(e)) out.push(e)
  }
  return out
}

export function rankPredictions(emojis: string[], vocab: Set<string>): string[] {
  const out: string[] = []
  for (const raw of emojis) {
    const e = stripVS(raw)
    if (vocab.has(e) && !out.includes(e)) out.push(e)
  }
  return out
}

export function hitAtK(preds: string[], targets: string[], k: number): boolean {
  return preds.slice(0, k).some((e) => targets.includes(e))
}

export function reciprocalRank(preds: string[], targets: string[]): number {
  const i = preds.findIndex((e) => targets.includes(e))
  return i === -1 ? 0 : 1 / (i + 1)
}

export type Row = { preds: string[]; targets: string[] }

export type Summary = {
  n: number
  accAtK: number[]
  mrr: number
  meanPreds: number
  zeroPredRows: number
}

export function summarize(rows: Row[]): Summary {
  const n = rows.length || 1
  return {
    n: rows.length,
    accAtK: KS.map(
      (k) => rows.filter((r) => hitAtK(r.preds, r.targets, k)).length / n,
    ),
    mrr: rows.reduce((s, r) => s + reciprocalRank(r.preds, r.targets), 0) / n,
    meanPreds: rows.reduce((s, r) => s + r.preds.length, 0) / n,
    zeroPredRows: rows.filter((r) => r.preds.length === 0).length,
  }
}

function buildIndex(docs: string[], tokenize: (typeof TOKENIZERS)[number]): Index {
  const index = new Index({ tokenize })
  docs.forEach((doc, id) => index.add(id, doc))
  return index
}

function pct(x: number): string {
  return (100 * x).toFixed(1).padStart(5)
}

if (import.meta.main) {
  const vocab = new Set<string>(
    (JSON.parse(await Bun.file(LABELS_JSON).text()).emojis as string[]).map(stripVS),
  )
  const annotations = await loadCldrAnnotations()
  const glyphs: string[] = []
  const docs: string[] = []
  for (const [glyph, keywords] of annotations) {
    glyphs.push(stripVS(glyph))
    docs.push(keywords.join(" "))
  }

  const evalRows = await readJsonl<{ text: string; emojis: string }>(EVAL_JSONL)
  const scored = evalRows
    .map((r) => ({ text: r.text, targets: rowTargets(String(r.emojis ?? ""), vocab) }))
    .filter((r) => r.targets.length > 0)

  console.log(
    `CLDR baseline: ${annotations.size} CLDR emoji indexed, `
    + `${scored.length}/${evalRows.length} eval rows scored `
    + `(rows with >=1 vocab emoji), predictions restricted to the ${vocab.size}-emoji vocab\n`,
  )

  const header = ["tokenizer", ...KS.map((k) => `acc@${k}`), "MRR@10", "pred/row", "0-pred"]
  console.log(header.map((h) => h.padStart(8)).join(" "))

  for (const tokenize of TOKENIZERS) {
    const index = buildIndex(docs, tokenize)
    const rows: Row[] = scored.map((r) => {
      const ids = index.search(r.text, { limit: SEARCH_LIMIT, suggest: true }) as number[]
      return { targets: r.targets, preds: rankPredictions(ids.map((id) => glyphs[id]), vocab) }
    })
    const s = summarize(rows)
    console.log(
      [
        tokenize.padStart(8),
        ...s.accAtK.map(pct),
        pct(s.mrr),
        s.meanPreds.toFixed(2).padStart(8),
        `${s.zeroPredRows}`.padStart(8),
      ].join(" "),
    )
  }
}
