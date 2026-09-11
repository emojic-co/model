import { describe, it, expect } from 'vitest'
import { fuse } from './fusion'

const emojiSigmoid = Float32Array.from([0.9, 0.5, 0.1, 0.7])
const kw = Float32Array.from([0, 0.8, 0, 0])

describe('fuse', () => {
  it('is a gate-weighted mix of the sigmoided model score and kw', () => {
    const gate = 0.3
    const out = fuse(gate, emojiSigmoid, kw)
    for (let i = 0; i < emojiSigmoid.length; i++) {
      expect(out[i]).toBeCloseTo(gate * emojiSigmoid[i] + (1 - gate) * kw[i], 5)
    }
  })

  it('gate=1 returns the model score untouched', () => {
    const out = fuse(1, emojiSigmoid, kw)
    for (let i = 0; i < emojiSigmoid.length; i++) {
      expect(out[i]).toBeCloseTo(emojiSigmoid[i], 5)
    }
  })

  it('gate=0 returns kw untouched', () => {
    const out = fuse(0, emojiSigmoid, kw)
    for (let i = 0; i < emojiSigmoid.length; i++) {
      expect(out[i]).toBeCloseTo(kw[i], 5)
    }
  })
})
