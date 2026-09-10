import { FUZZY_MIN_LEN, queryTokens } from "../analysis/cldr-baseline.ts"
import { MIN_FUZZY_SCORE } from "./config"

export type FlexJson = { kw_vocab: string[] }

const r3 = (x: number) => Number(x.toFixed(3))

function overlap(w: string, k: string): number {
  if (w === k) return 1.0
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0.0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0.0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0.0
}

export function buildFlexRanker(kwVocab: string[]): {
  kwVocab: string[]
  tfVec: (text: string) => number[]
  buildJson: () => FlexJson
} {
  const vocab = [...kwVocab]

  const tfVec = (text: string): number[] => {
    const q = queryTokens(text)
    return vocab.map((k) => {
      let s = 0
      for (const w of q) s += overlap(w, k)
      return r3(s)
    })
  }

  return { kwVocab: vocab, tfVec, buildJson: () => ({ kw_vocab: vocab }) }
}
