import { Index } from "flexsearch"

import { EVAL_JSONL, LABELS_JSON } from "../../files.ts"
import { readJsonl } from "../data/io.ts"
import { loadCldrAnnotations } from "../data/cldr.ts"

const VARIATION_SELECTORS = /[︎️]/g
const SEARCH_LIMIT = 50
const KS = Array.from({ length: 10 }, (_, i) => i + 1)
const TOKENIZERS = ["strict", "forward"] as const
const FUZZY_MIN_LEN = 4
const FUZZY_MAX_LEN_DELTA = 3
const FUZZY_WEIGHT = 0.6

const STOPWORDS = new Set(
  ("a an the to of in on at is it its i you we they he she this that for and or but"
    + " not with my your me am are was were be been being do does did have has had"
    + " will would can could just so if").split(" "),
)

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

export function queryTokens(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w))
}

export function fuzzyMatch(a: string, b: string): boolean {
  if (a.length < FUZZY_MIN_LEN || b.length < FUZZY_MIN_LEN) return false
  if (Math.abs(a.length - b.length) > FUZZY_MAX_LEN_DELTA) return false
  return a.startsWith(b) || b.startsWith(a)
}

export function makeIdf(docs: string[][]): (word: string) => number {
  const df = new Map<string, number>()
  for (const doc of docs) {
    for (const w of new Set(doc)) df.set(w, (df.get(w) ?? 0) + 1)
  }
  const n = docs.length
  return (word) => Math.log((n + 1) / ((df.get(word) ?? 0) + 1)) + 1
}

export function overlapRank(
  qTokens: string[],
  docs: string[][],
  glyphs: string[],
  idf: (word: string) => number,
): string[] {
  const scored: { glyph: string; score: number; i: number }[] = []
  for (let i = 0; i < docs.length; i++) {
    const kw = new Set(docs[i])
    let score = 0
    for (const w of qTokens) {
      if (kw.has(w)) {
        score += idf(w)
      } else if ([...kw].some((k) => fuzzyMatch(w, k))) {
        score += idf(w) * FUZZY_WEIGHT
      }
    }
    if (score > 0) scored.push({ glyph: glyphs[i], score, i })
  }
  scored.sort((a, b) => b.score - a.score || a.i - b.i)
  return scored.map((s) => s.glyph)
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
  return (100 * x).toFixed(1).padStart(9)
}

async function evaluate() {
  const vocab = new Set<string>(
    (JSON.parse(await Bun.file(LABELS_JSON).text()).emojis as string[]).map(stripVS),
  )
  const annotations = await loadCldrAnnotations()
  const glyphs: string[] = []
  const docs: string[] = []
  const kwTokens: string[][] = []
  for (const [glyph, keywords] of annotations) {
    glyphs.push(stripVS(glyph))
    docs.push(keywords.join(" "))
    kwTokens.push(keywords.map((k) => k.toLowerCase()))
  }
  const idf = makeIdf(kwTokens)

  const evalRows = await readJsonl<{ text: string; emojis: string }>(EVAL_JSONL)
  const scored = evalRows
    .map((r) => ({ text: r.text, targets: rowTargets(String(r.emojis ?? ""), vocab) }))
    .filter((r) => r.targets.length > 0)

  const methods: Record<string, Row[]> = {}
  for (const tokenize of TOKENIZERS) {
    const index = buildIndex(docs, tokenize)
    methods[tokenize] = scored.map((r) => {
      const ids = index.search(r.text, { limit: SEARCH_LIMIT, suggest: true }) as number[]
      return { targets: r.targets, preds: rankPredictions(ids.map((id) => glyphs[id]), vocab) }
    })
  }
  methods.overlap = scored.map((r) => ({
    targets: r.targets,
    preds: rankPredictions(overlapRank(queryTokens(r.text), kwTokens, glyphs, idf), vocab),
  }))

  return {
    cldrEmoji: annotations.size,
    vocab: vocab.size,
    scored: scored.length,
    total: evalRows.length,
    summaries: Object.fromEntries(
      Object.entries(methods).map(([name, rows]) => [name, summarize(rows)]),
    ),
  }
}

export async function runBaseline() {
  const { cldrEmoji, vocab, scored, total, summaries } = await evaluate()
  return {
    cldr_emoji: cldrEmoji,
    vocab,
    scored,
    total,
    methods: Object.fromEntries(
      Object.entries(summaries).map(([name, s]) => [
        name,
        {
          acc_at_k: s.accAtK,
          mrr: s.mrr,
          mean_preds: s.meanPreds,
          zero_pred_rows: s.zeroPredRows,
        },
      ]),
    ),
  }
}

if (import.meta.main) {
  const asJson = process.argv.includes("--json")

  if (asJson) {
    console.log(JSON.stringify(await runBaseline(), null, 2))
  } else {
    const { cldrEmoji, vocab, scored, total, summaries } = await evaluate()
    console.log(
      `CLDR baseline: ${cldrEmoji} CLDR emoji indexed, ${scored}/${total} eval rows scored `
      + `(rows with >=1 vocab emoji), predictions restricted to the ${vocab}-emoji vocab\n`,
    )
    const header = ["method", ...KS.map((k) => `acc@${k}`), "MRR@10", "pred/row", "0-pred"]
    console.log(header.map((h) => h.padStart(9)).join(" "))
    for (const [name, s] of Object.entries(summaries)) {
      console.log(
        [
          name.padStart(9),
          ...s.accAtK.map(pct),
          pct(s.mrr),
          s.meanPreds.toFixed(2).padStart(9),
          `${s.zeroPredRows}`.padStart(9),
        ].join(" "),
      )
    }
  }
}
