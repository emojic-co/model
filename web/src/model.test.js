import { describe, it, expect } from 'vitest'
import {
  normalize,
  encode,
  argmax,
  softmax,
  oklabToHex,
  decodeColors,
  paletteVariants,
} from './model'

const CHARS = '·abcdefghijklmnopqrstuvwxyz!?:()@$%&* '
const idx = new Map([...CHARS].map((c, i) => [c, i]))

describe('normalize', () => {
  it('lowercases and collapses whitespace', () => {
    expect(normalize('Hello   WORLD', idx)).toBe('hello world')
  })
  it('collapses 3+ char repeats to 2', () => {
    expect(normalize('soooo good', idx)).toBe('soo good')
  })
  it('drops chars outside the vocab (incl. digits and accents)', () => {
    expect(normalize('café #1!', idx)).toBe('caf !')
  })
  it('trims leading/trailing whitespace', () => {
    expect(normalize('  hi there  ', idx)).toBe('hi there')
  })
})

describe('encode', () => {
  const meta = { max_text_len: 5, pad_idx: 0 }
  it('maps chars to indices and pads to max_text_len', () => {
    expect(Array.from(encode('ab', meta, idx))).toEqual([1n, 2n, 0n, 0n, 0n])
  })
  it('truncates to max_text_len', () => {
    expect(Array.from(encode('abcdef', meta, idx))).toEqual([1n, 2n, 3n, 4n, 5n])
  })
  it('returns a BigInt64Array', () => {
    expect(encode('a', meta, idx)).toBeInstanceOf(BigInt64Array)
  })
})

describe('oklabToHex', () => {
  const cases = [
    ['#b4000f', [0.484144, 0.175408, 0.090798]],
    ['#ffd571', [0.889335, 0.006905, 0.127403]],
    ['#2f1100', [0.220989, 0.036459, 0.044766]],
    ['#5bd9a4', [0.800294, -0.130398, 0.040409]],
    ['#78c9f4', [0.800108, -0.060116, -0.080137]],
    ['#ffffff', [1, 0, 0]],
    ['#000000', [0, 0, 0]],
  ]
  for (const [hex, lab] of cases) {
    it(`round-trips ${hex}`, () => {
      expect(oklabToHex(lab[0], lab[1], lab[2])).toBe(hex)
    })
  }
  it('always returns a valid hex, even for out-of-gamut input', () => {
    expect(oklabToHex(2, 1, 1)).toMatch(/^#[0-9a-f]{6}$/)
    expect(oklabToHex(-1, -1, -1)).toMatch(/^#[0-9a-f]{6}$/)
    expect(oklabToHex(0.5, 5, -5)).toMatch(/^#[0-9a-f]{6}$/)
  })
})

describe('decodeColors', () => {
  it('splits the 9-vector into bg1/bg2/text_color', () => {
    const c = [
      0.484144, 0.175408, 0.090798, 0.889335, 0.006905, 0.127403, 0.220989,
      0.036459, 0.044766,
    ]
    expect(decodeColors(c)).toEqual({
      bg1: '#b4000f',
      bg2: '#ffd571',
      text_color: '#2f1100',
    })
  })
})

describe('paletteVariants', () => {
  const color = [
    0.484144, 0.175408, 0.090798, 0.889335, 0.006905, 0.127403, 0.220989,
    0.036459, 0.044766,
  ]
  const hex = /^#[0-9a-f]{6}$/

  it('returns the model palette plus n variants', () => {
    expect(paletteVariants(color)).toHaveLength(5)
    expect(paletteVariants(color, 2)).toHaveLength(3)
  })

  it('keeps the model prediction as the first entry', () => {
    expect(paletteVariants(color)[0]).toEqual(decodeColors(color))
  })

  it('every entry is three valid hex colors', () => {
    for (const p of paletteVariants(color)) {
      expect(p.bg1).toMatch(hex)
      expect(p.bg2).toMatch(hex)
      expect(p.text_color).toMatch(hex)
    }
  })

  it('each variant differs from the model prediction', () => {
    const [base, ...variants] = paletteVariants(color)
    for (const v of variants) expect(v).not.toEqual(base)
  })

  it('is deterministic for the same input', () => {
    expect(paletteVariants(color)).toEqual(paletteVariants(color))
  })
})

describe('argmax / softmax', () => {
  it('argmax returns the index of the max', () => {
    expect(argmax([0.1, 0.9, 0.3])).toBe(1)
  })
  it('softmax sums to 1 and is monotonic', () => {
    const p = softmax([1, 2, 3])
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
    expect(p[2]).toBeGreaterThan(p[0])
  })
})
