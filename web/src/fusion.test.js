import { describe, it, expect } from 'vitest'
import { makeFusion } from './fusion'

const logits = Float32Array.from([2, 0, -1, 1])
const kw = Float32Array.from([0, 0.8, 0, 0])

function zscore(a) {
  const m = a.reduce((s, x) => s + x, 0) / a.length
  const v = a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length
  return a.map((x) => (x - m) / (Math.sqrt(v) + 1e-6))
}

describe('makeFusion', () => {
  it('gain: fused = logits + softplus(beta) * kw', () => {
    const f = makeFusion({ variant: 'gain', beta: 0 })
    const out = f.fuse(logits, kw)
    const sp = Math.log1p(Math.exp(0))
    expect(out[1]).toBeCloseTo(logits[1] + sp * kw[1], 5)
    expect(out[0]).toBeCloseTo(logits[0], 5)
  })

  it('mix: fused = (1-g) z(logits) + g kw', () => {
    const f = makeFusion({ variant: 'mix', g: 0.25 })
    const z = zscore(Array.from(logits))
    const out = f.fuse(logits, kw)
    expect(out[1]).toBeCloseTo(0.75 * z[1] + 0.25 * kw[1], 4)
  })

  it('gate: a in (0,1), kw=0 entries equal a*z(logits)', () => {
    const f = makeFusion({
      variant: 'gate',
      bn_mean: [0, 0, 0, 0, 0, 0],
      bn_var: [1, 1, 1, 1, 1, 1],
      w: [0, 0, 0, 0, 0, 0],
      b: 0,
    })
    const z = zscore(Array.from(logits))
    const out = f.fuse(logits, kw)
    expect(out[0]).toBeCloseTo(0.5 * z[0], 4)
  })
})
