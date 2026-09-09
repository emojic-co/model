import { FUZZY_MIN_LEN, queryTokens } from "../analysis/cldr-baseline.ts"
import { loadCldrAnnotations } from "./cldr.ts"
import { KW_MAX_LEN, KW_MIN_LEN, MIN_FUZZY_SCORE } from "./config"

export type FlexJson = { kw_vocab: string[] }

const r3 = (x: number) => Number(x.toFixed(3))

function overlap(w: string, k: string): number {
  if (w === k) return 1.0
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0.0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0.0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0.0
}

export async function buildFlexRanker(corpusTexts: string[]): Promise<{
  kwVocab: string[]
  tfVec: (text: string) => number[]
  buildJson: () => FlexJson
}> {
  const annotations = await loadCldrAnnotations()
  const df = new Map<string, number>()
  for (const [, keywords] of annotations) {
    const uniq = new Set(
      keywords.map((w) => w.trim().toLowerCase()).filter(Boolean),
    )
    for (const kw of uniq) df.set(kw, (df.get(kw) ?? 0) + 1)
  }

  const corpusTokens = new Set<string>()
  for (const t of corpusTexts) for (const w of queryTokens(t)) corpusTokens.add(w)

  const kwVocab: string[] = []
  for (const [kw, n] of df) {
    if (n !== 1) continue
    if (kw.length < KW_MIN_LEN || kw.length > KW_MAX_LEN) continue
    const qt = queryTokens(kw)
    if (qt.length !== 1 || qt[0] !== kw) continue
    if (!corpusTokens.has(kw)) continue
    kwVocab.push(kw)
  }
  kwVocab.sort()

  const tfVec = (text: string): number[] => {
    const q = queryTokens(text)
    return kwVocab.map((k) => {
      let s = 0
      for (const w of q) s += overlap(w, k)
      return r3(s)
    })
  }

  return { kwVocab, tfVec, buildJson: () => ({ kw_vocab: kwVocab }) }
}
