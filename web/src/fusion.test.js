import { describe, it, expect } from 'vitest'
import { makeFusion } from './fusion'

const logits = Float32Array.from([2, 0, -1, 1])
const kw = Float32Array.from([0, 0.8, 0, 0])

describe('makeFusion', () => {
  it('fused = w_dl * logits + w_search * kw + b (per emoji)', () => {
    const f = makeFusion({
      w_dl: [1, 1, 1, 1],
      w_search: [1, 1, 1, 1],
      b: [0, 0, 0, 0],
    })
    const out = f.fuse(logits, kw)
    for (let i = 0; i < logits.length; i++) {
      expect(out[i]).toBeCloseTo(logits[i] + kw[i], 5)
    }
  })

  it('honours per-emoji weights and bias', () => {
    const f = makeFusion({
      w_dl: [0.5, 2, 1, 0],
      w_search: [0, 3, 1, 1],
      b: [0.1, -0.2, 0, 1],
    })
    const out = f.fuse(logits, kw)
    expect(out[0]).toBeCloseTo(0.5 * 2 + 0 * 0 + 0.1, 5)
    expect(out[1]).toBeCloseTo(2 * 0 + 3 * 0.8 - 0.2, 5)
    expect(out[2]).toBeCloseTo(1 * -1 + 1 * 0 + 0, 5)
    expect(out[3]).toBeCloseTo(0 * 1 + 1 * 0 + 1, 5)
  })
})
