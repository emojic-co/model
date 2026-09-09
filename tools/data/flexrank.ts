import {
  FUZZY_MIN_LEN,
  FUZZY_WEIGHT,
  fuzzyMatch,
  makeIdf,
  queryTokens,
  stripVS,
} from "../analysis/cldr-baseline.ts"
import { loadCldrAnnotations } from "./cldr.ts"

export const FLEX_COLS = [
  "emoji",
  "score",
  "score_norm",
  "exact",
  "fuzzy",
  "best_idf",
  "n_kw",
  "kw_len",
  "word_len",
  "overlap",
] as const
export type FlexHit = [
  emoji: string,
  score: number,
  score_norm: number,
  exact: number,
  fuzzy: number,
  best_idf: number,
  n_kw: number,
  kw_len: number,
  word_len: number,
  overlap: number,
]
export type FlexQ = {
  tokens: number
  matched: number
  sum: number
  max: number
  cand: number
}
export type FlexResult = { flexsearch: FlexHit[]; flexq: FlexQ }
export type FlexRanker = (text: string, vocab: Map<string, string>) => FlexResult
export type FlexJson = {
  emojis: string[]
  keywords: string[][]
  idf: Record<string, number>
  idf_default: number
}
export type FlexRankerBundle = {
  rank: FlexRanker
  buildJson: (vocabList: string[]) => FlexJson
}

const r3 = (x: number) => Number(x.toFixed(3))

export async function buildFlexRanker(k: number): Promise<FlexRankerBundle> {
  const annotations = await loadCldrAnnotations()
  const glyphs: string[] = []
  const kwCount: number[] = []
  const kwTokens: string[][] = []
  const kwToGlyphs = new Map<string, Set<number>>()
  const prefix4 = new Map<string, Set<string>>()
  for (const [glyph, keywords] of annotations) {
    const gi = glyphs.length
    glyphs.push(stripVS(glyph))
    const toks = keywords.map((w) => w.toLowerCase())
    kwTokens.push(toks)
    kwCount.push(toks.length)
    for (const kw of toks) {
      let gs = kwToGlyphs.get(kw)
      if (!gs) kwToGlyphs.set(kw, (gs = new Set()))
      gs.add(gi)
      if (kw.length >= FUZZY_MIN_LEN) {
        const p = kw.slice(0, FUZZY_MIN_LEN)
        let bucket = prefix4.get(p)
        if (!bucket) prefix4.set(p, (bucket = new Set()))
        bucket.add(kw)
      }
    }
  }
  const idf = makeIdf(kwTokens)

  let memoVocab: Map<string, string> | null = null
  let memoVocabKw = new Set<string>()
  let memoVocabRank = new Map<string, number>()
  const ensureVocab = (vocab: Map<string, string>) => {
    if (vocab === memoVocab) return
    memoVocab = vocab
    memoVocabKw = new Set()
    memoVocabRank = new Map()
    let pos = 0
    for (const stripped of vocab.keys()) memoVocabRank.set(stripped, pos++)
    for (let gi = 0; gi < glyphs.length; gi++) {
      if (!vocab.has(glyphs[gi])) continue
      for (const kw of kwTokens[gi]) {
        for (const w of kw.split(/\s+/)) if (w) memoVocabKw.add(w)
      }
    }
  }

  const rank: FlexRanker = (text, vocab) => {
    ensureVocab(vocab)
    const qTokens = queryTokens(text)
    type Acc = {
      score: number
      exact: number
      fuzzy: number
      bestIdf: number
      bestExact: boolean
      kwLen: number
      wordLen: number
      overlap: number
    }
    const acc = new Map<number, Acc>()
    const bump = (
      gi: number,
      s: number,
      tokIdf: number,
      isExact: boolean,
      wordLen: number,
      kwLen: number,
      overlap: number,
    ) => {
      if (!vocab.has(glyphs[gi])) return
      let e = acc.get(gi)
      if (!e) {
        e = {
          score: 0,
          exact: 0,
          fuzzy: 0,
          bestIdf: -1,
          bestExact: false,
          kwLen: 0,
          wordLen: 0,
          overlap: 0,
        }
        acc.set(gi, e)
      }
      e.score += s
      if (isExact) e.exact += 1
      else e.fuzzy += 1
      const win =
        tokIdf > e.bestIdf ||
        (tokIdf === e.bestIdf && isExact && !e.bestExact) ||
        (tokIdf === e.bestIdf && isExact === e.bestExact && overlap > e.overlap)
      if (win) {
        e.bestIdf = tokIdf
        e.bestExact = isExact
        e.kwLen = kwLen
        e.wordLen = wordLen
        e.overlap = overlap
      }
    }
    for (const w of qTokens) {
      const wIdf = idf(w)
      const exact = kwToGlyphs.get(w)
      if (exact) for (const gi of exact) bump(gi, wIdf, wIdf, true, w.length, w.length, w.length)
      if (w.length >= FUZZY_MIN_LEN) {
        const bucket = prefix4.get(w.slice(0, FUZZY_MIN_LEN))
        if (bucket) {
          const fuzz = new Map<number, { kw: string; ov: number }>()
          for (const kw of bucket) {
            if (kw === w || !fuzzyMatch(w, kw)) continue
            const ov = Math.min(w.length, kw.length)
            for (const gi of kwToGlyphs.get(kw)!) {
              const cur = fuzz.get(gi)
              if (!cur || ov > cur.ov) fuzz.set(gi, { kw, ov })
            }
          }
          for (const [gi, m] of fuzz) {
            if (exact?.has(gi)) continue
            bump(gi, wIdf * FUZZY_WEIGHT, wIdf, false, w.length, m.kw.length, m.ov)
          }
        }
      }
    }

    let sum = 0
    let max = 0
    for (const v of acc.values()) {
      sum += v.score
      if (v.score > max) max = v.score
    }
    const vrank = (gi: number) => memoVocabRank.get(glyphs[gi]) ?? 0
    const ranked = [...acc.entries()].sort(
      (a, b) =>
        b[1].score - a[1].score ||
        (b[1].exact + b[1].fuzzy) - (a[1].exact + a[1].fuzzy) ||
        vrank(a[0]) - vrank(b[0]),
    )
    const list: FlexHit[] = []
    for (const [gi, v] of ranked) {
      const orig = vocab.get(glyphs[gi])!
      list.push([
        orig,
        r3(v.score),
        max ? r3(v.score / max) : 0,
        v.exact,
        v.fuzzy,
        r3(v.bestIdf),
        kwCount[gi],
        v.kwLen,
        v.wordLen,
        v.overlap,
      ])
      if (list.length >= k) break
    }
    const matched = qTokens.filter((w) => memoVocabKw.has(w)).length
    return {
      flexsearch: list,
      flexq: { tokens: qTokens.length, matched, sum: r3(sum), max: r3(max), cand: acc.size },
    }
  }

  const allKeywords = new Set<string>()
  for (const toks of kwTokens) for (const kw of toks) allKeywords.add(kw)
  const idfMap: Record<string, number> = {}
  for (const kw of allKeywords) idfMap[kw] = idf(kw)
  const idfDefault = Math.log(glyphs.length + 1) + 1
  const glyphToIdx = new Map(glyphs.map((g, i) => [g, i]))

  const buildJson = (vocabList: string[]): FlexJson => {
    const emojis: string[] = []
    const keywords: string[][] = []
    for (const orig of vocabList) {
      const gi = glyphToIdx.get(stripVS(orig))
      emojis.push(orig)
      keywords.push(gi === undefined ? [] : kwTokens[gi])
    }
    return { emojis, keywords, idf: idfMap, idf_default: idfDefault }
  }

  return { rank, buildJson }
}
