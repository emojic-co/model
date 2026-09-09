import { describe, expect, it } from 'vitest'
import flexJson from '../public/flex.json'
import fixture from './flexrank.fixture.json'
import { makeFlexRanker } from './flexrank'

describe('flexrank parity', () => {
  const r = makeFlexRanker(flexJson)

  it('kw_vocab matches fixture', () => {
    expect(r.kwVocab).toEqual(fixture.kw_vocab)
  })

  for (const c of fixture.cases) {
    it(`tfVec matches fixture: ${JSON.stringify(c.text)}`, () => {
      const got = r.tfVec(c.text)
      expect(got.length).toBe(c.tf.length)
      got.forEach((v, i) => expect(v).toBeCloseTo(c.tf[i], 6))
    })
  }
})
