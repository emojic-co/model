const FUZZY_MIN_LEN = 4
const MIN_FUZZY_SCORE = 0.66
const STOPWORDS = new Set(
  (
    'a an the to of in on at is it its i you we they he she this that for and or but' +
    ' not with my your me am are was were be been being do does did have has had' +
    ' will would can could just so if'
  ).split(' '),
)
const r3 = (x) => Number(x.toFixed(3))

function queryTokens(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length >= 2 && !STOPWORDS.has(w))
}

function overlap(w, k) {
  if (w === k) return 1
  if (w.length < FUZZY_MIN_LEN || k.length < FUZZY_MIN_LEN) return 0
  if (!(w.startsWith(k) || k.startsWith(w))) return 0
  const r = Math.min(w.length, k.length) / Math.max(w.length, k.length)
  return r >= MIN_FUZZY_SCORE ? r : 0
}

export function makeFlexRanker(flexJson) {
  const kwVocab = flexJson.kw_vocab
  return {
    kwVocab,
    tfVec(text) {
      const q = queryTokens(text)
      return kwVocab.map((k) => {
        let s = 0
        for (const w of q) s += overlap(w, k)
        return r3(s)
      })
    },
  }
}
