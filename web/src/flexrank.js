const FUZZY_MIN_LEN = 4
const FUZZY_MAX_LEN_DELTA = 3
const FUZZY_WEIGHT = 0.6
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

function fuzzyMatch(a, b) {
  if (a.length < FUZZY_MIN_LEN || b.length < FUZZY_MIN_LEN) return false
  if (Math.abs(a.length - b.length) > FUZZY_MAX_LEN_DELTA) return false
  return a.startsWith(b) || b.startsWith(a)
}

export function makeFlexRanker(flexJson) {
  const emojis = flexJson.emojis
  const keywords = flexJson.keywords
  const idfMap = flexJson.idf
  const idfDefault = flexJson.idf_default
  const kwCount = keywords.map((k) => k.length)
  const kwToGlyphs = new Map()
  const prefix4 = new Map()
  const globalKw = new Set()
  keywords.forEach((toks, gi) => {
    for (const kw of toks) {
      for (const w of kw.split(/\s+/)) if (w) globalKw.add(w)
      let s = kwToGlyphs.get(kw)
      if (!s) kwToGlyphs.set(kw, (s = new Set()))
      s.add(gi)
      if (kw.length >= FUZZY_MIN_LEN) {
        const p = kw.slice(0, FUZZY_MIN_LEN)
        let b = prefix4.get(p)
        if (!b) prefix4.set(p, (b = new Set()))
        b.add(kw)
      }
    }
  })
  const idf = (w) => (w in idfMap ? idfMap[w] : idfDefault)

  function scoreAll(text) {
    const q = queryTokens(text)
    const acc = new Map()
    const bump = (gi, s, tokIdf, isExact, wLen, kLen, ov) => {
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
        (tokIdf === e.bestIdf && isExact === e.bestExact && ov > e.overlap)
      if (win) {
        e.bestIdf = tokIdf
        e.bestExact = isExact
        e.kwLen = kLen
        e.wordLen = wLen
        e.overlap = ov
      }
    }
    for (const w of q) {
      const wIdf = idf(w)
      const exact = kwToGlyphs.get(w)
      if (exact) for (const gi of exact) bump(gi, wIdf, wIdf, true, w.length, w.length, w.length)
      if (w.length >= FUZZY_MIN_LEN) {
        const bucket = prefix4.get(w.slice(0, FUZZY_MIN_LEN))
        if (bucket) {
          const fuzz = new Map()
          for (const kw of bucket) {
            if (kw === w || !fuzzyMatch(w, kw)) continue
            const ov = Math.min(w.length, kw.length)
            for (const gi of kwToGlyphs.get(kw)) {
              const cur = fuzz.get(gi)
              if (!cur || ov > cur.ov) fuzz.set(gi, { kw, ov })
            }
          }
          for (const [gi, m] of fuzz) {
            if (exact && exact.has(gi)) continue
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
    const matched = q.filter((w) => globalKw.has(w)).length
    return { q, acc, sum, max, matched }
  }

  function rankedEntries(text) {
    const { acc, max } = scoreAll(text)
    return [...acc.entries()]
      .sort(
        (a, b) =>
          b[1].score - a[1].score ||
          b[1].exact + b[1].fuzzy - (a[1].exact + a[1].fuzzy) ||
          a[0] - b[0],
      )
      .slice(0, 32)
      .map(([gi, v], pos) => ({ gi, v, pos, max }))
  }

  return {
    rank(text) {
      return rankedEntries(text).map(({ gi, v, max }) => [
        emojis[gi],
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
    },
    flexRaw(text) {
      const out = new Float32Array(emojis.length * 10)
      for (const { gi, v, pos, max } of rankedEntries(text)) {
        const o = gi * 10
        out[o + 0] = r3(v.score)
        out[o + 1] = max ? r3(v.score / max) : 0
        out[o + 2] = v.exact
        out[o + 3] = v.fuzzy
        out[o + 4] = r3(v.bestIdf)
        out[o + 5] = 1 / (pos + 1)
        out[o + 6] = kwCount[gi]
        out[o + 7] = v.kwLen
        out[o + 8] = v.wordLen
        out[o + 9] = v.overlap
      }
      return out
    },
    flexQ(text) {
      const { q, acc, sum, max, matched } = scoreAll(text)
      return [q.length, matched, r3(sum), r3(max), acc.size]
    },
  }
}
