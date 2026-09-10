import uFuzzy from '@leeoniya/ufuzzy'
import { queryTokens } from './tokenize'

export function makeKeywordPredictor(kwprojJson, emojiCount) {
  const proj = kwprojJson.proj
  const keys = Object.keys(proj)
  const weight = new Map(keys.map((k) => [k, 1 / Math.log2(1 + proj[k].length)]))
  const uf = new uFuzzy({ intraIns: 1 })
  function* matches(word) {
    if (proj[word]) yield [word, 1.0]
    if (word.length < 3) return
    const idxs = uf.filter(keys, word)
    if (!idxs || !idxs.length) return
    const info = uf.info(idxs, keys, word)
    for (let i = 0; i < info.idx.length; i++) {
      const k = keys[info.idx[i]]
      if (k === word) continue
      const sim = info.chars[i] / k.length
      if (sim >= 0.5) yield [k, sim]
    }
  }
  return {
    predict(text) {
      const out = new Float32Array(emojiCount)
      for (const word of queryTokens(text)) {
        for (const [k, strength] of matches(word)) {
          const v = strength * weight.get(k)
          for (const e of proj[k]) if (v > out[e]) out[e] = v
        }
      }
      return out
    },
  }
}
