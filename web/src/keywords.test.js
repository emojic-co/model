import { describe, it, expect } from 'vitest'
import { makeKeywordPredictor } from './keywords'

const proj = {
  pizza: [0],
  beaches: [1, 2],
  dog: [3],
}

describe('makeKeywordPredictor', () => {
  it('scores an exact keyword hit above zero, others zero', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    const v = kp.predict('pizza tonight')
    expect(v.length).toBe(4)
    expect(v[0]).toBeGreaterThan(0)
    expect(v[3]).toBe(0)
  })

  it('down-weights keys that fan out to many emoji', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    const v = kp.predict('a day at the beaches')
    expect(v[1]).toBeGreaterThan(0)
    expect(v[1]).toBeLessThan(1)
  })

  it('drops stopwords / short words before matching', () => {
    const kp = makeKeywordPredictor({ proj: { dog: [3], the: [0] } }, 4)
    const v = kp.predict('the the the')
    expect(Array.from(v)).toEqual([0, 0, 0, 0])
  })

  it('returns all zeros when nothing matches', () => {
    const kp = makeKeywordPredictor({ proj }, 4)
    expect(Array.from(kp.predict('qwerty xyzzy'))).toEqual([0, 0, 0, 0])
  })
})
