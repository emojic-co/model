import { describe, it, expect } from 'vitest'
import fixture from './flexrank.fixture.json'
import flexJson from '../public/flex.json'
import { makeFlexRanker } from './flexrank'

describe('flexrank parity', () => {
  const r = makeFlexRanker(flexJson)

  for (const c of fixture.cases) {
    it(`matches fixture: ${JSON.stringify(c.text)}`, () => {
      const got = r.rank(c.text)
      expect(got.length).toBe(c.flexsearch.length)
      got.forEach((row, i) => {
        const exp = c.flexsearch[i]
        expect(row[0]).toBe(exp[0])
        for (let k = 1; k < row.length; k++) expect(row[k]).toBeCloseTo(exp[k], 6)
      })
      const q = r.flexQ(c.text)
      expect(q[0]).toBe(c.flexq.tokens)
      expect(q[1]).toBe(c.flexq.matched)
      expect(q[2]).toBeCloseTo(c.flexq.sum, 6)
      expect(q[3]).toBeCloseTo(c.flexq.max, 6)
      expect(q[4]).toBe(c.flexq.cand)
    })
  }

  it('flexRaw is V*10 and puts rank_recip after best_idf', () => {
    const raw = r.flexRaw('pizza time with friends tonight')
    expect(raw.length).toBe(flexJson.emojis.length * 10)
    const top = r.rank('pizza time with friends tonight')[0][0]
    const ti = flexJson.emojis.indexOf(top)
    expect(raw[ti * 10 + 5]).toBeCloseTo(1, 6)
  })
})
