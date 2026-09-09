import { describe, it, expect } from 'vitest'
import {
  normalize,
  encode,
  argmax,
  softmax,
  sigmoid,
  decodeColors,
  decodeColorList,
  srgbToOklab,
  oklabToSrgb,
  contrastRatio,
  fixContrast,
  CONTRAST_MIN,
} from './model'

const CHARS = '·abcdefghijklmnopqrstuvwxyz0123456789!?:()@$%&* '
const idx = new Map([...CHARS].map((c, i) => [c, i]))

describe('normalize', () => {
  it('lowercases and collapses whitespace', () => {
    expect(normalize('Hello   WORLD', idx)).toBe('hello world')
  })
  it('collapses 3+ char repeats to 2', () => {
    expect(normalize('soooo good', idx)).toBe('soo good')
  })
  it('drops chars outside the vocab (incl. accents) but keeps digits', () => {
    expect(normalize('café #1!', idx)).toBe('caf 1!')
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

describe('decodeColors', () => {
  it('splits a 9-vector of 0..255 values into bg1/bg2/text_color hex', () => {
    expect(decodeColors([255, 0, 0, 0, 127.5, 255, 16, 16, 16])).toEqual({
      bg1: '#ff0000',
      bg2: '#0080ff',
      text_color: '#101010',
    })
  })
  it('clamps values outside 0..255', () => {
    expect(decodeColors([300, -5, 128, 300, 300, 300, 0, 0, 0])).toEqual({
      bg1: '#ff0080',
      bg2: '#ffffff',
      text_color: '#000000',
    })
  })
})

describe('decodeColorList', () => {
  const hex = /^#[0-9a-f]{6}$/

  it('chunks a flat 45-value buffer into 5 palettes', () => {
    const list = decodeColorList(new Float32Array(45).fill(0))
    expect(list).toHaveLength(5)
    for (const p of list) {
      expect(p.bg1).toMatch(hex)
      expect(p.bg2).toMatch(hex)
      expect(p.text_color).toMatch(hex)
    }
  })

  it('decodes each chunk independently', () => {
    const flat = [...Array(9).fill(0), ...Array(9).fill(255)]
    expect(decodeColorList(flat)).toEqual([
      { bg1: '#000000', bg2: '#000000', text_color: '#000000' },
      { bg1: '#ffffff', bg2: '#ffffff', text_color: '#ffffff' },
    ])
  })
})

describe('oklab', () => {
  it('round-trips sRGB through OKLab', () => {
    for (const rgb of [
      [255, 255, 255],
      [0, 0, 0],
      [128, 64, 200],
      [20, 180, 90],
    ]) {
      const back = oklabToSrgb(srgbToOklab(rgb)).map(Math.round)
      expect(back).toEqual(rgb)
    }
  })
  it('white is L≈1 with near-zero a/b', () => {
    const [L, a, b] = srgbToOklab([255, 255, 255])
    expect(L).toBeCloseTo(1, 3)
    expect(a).toBeCloseTo(0, 3)
    expect(b).toBeCloseTo(0, 3)
  })
})

describe('contrastRatio', () => {
  it('is 21 for black on white and 1 for a colour on itself', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5)
    expect(contrastRatio('#3a7bd5', '#3a7bd5')).toBeCloseTo(1, 5)
  })
  it('is symmetric', () => {
    expect(contrastRatio('#123456', '#abcdef')).toBeCloseTo(
      contrastRatio('#abcdef', '#123456'),
      10,
    )
  })
})

describe('fixContrast', () => {
  it('leaves a readable palette untouched (same object)', () => {
    const p = { bg1: '#a8e2f4', bg2: '#78c9f4', text_color: '#282e36' }
    expect(fixContrast(p)).toBe(p)
  })

  it('repairs a low-contrast palette so both stops clear the threshold', () => {
    const p = { bg1: '#2b2b2b', bg2: '#3a3a3a', text_color: '#444444' }
    const fixed = fixContrast(p)
    expect(fixed.text_color).not.toBe(p.text_color)
    expect(fixed.bg1).toBe(p.bg1)
    expect(fixed.bg2).toBe(p.bg2)
    expect(contrastRatio(fixed.text_color, fixed.bg1)).toBeGreaterThanOrEqual(CONTRAST_MIN)
    expect(contrastRatio(fixed.text_color, fixed.bg2)).toBeGreaterThanOrEqual(CONTRAST_MIN)
  })

  it('nudges lightness while roughly preserving hue', () => {
    const p = { bg1: '#c94f4f', bg2: '#d46b4b', text_color: '#b84a3a' }
    const fixed = fixContrast(p)
    const [, a0, b0] = srgbToOklab([0xb8, 0x4a, 0x3a])
    const n = parseInt(fixed.text_color.slice(1), 16)
    const [, a1, b1] = srgbToOklab([(n >> 16) & 255, (n >> 8) & 255, n & 255])
    expect(Math.sign(a1)).toBe(Math.sign(a0))
    expect(Math.sign(b1)).toBe(Math.sign(b0))
  })

  it('respects a custom threshold', () => {
    const p = { bg1: '#ffffff', bg2: '#f4f4f4', text_color: '#8a8a8a' }
    const fixed = fixContrast(p, 4.5)
    expect(minOf(fixed)).toBeGreaterThanOrEqual(4.5)
  })
})

function minOf({ bg1, bg2, text_color }) {
  return Math.min(contrastRatio(text_color, bg1), contrastRatio(text_color, bg2))
}

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

describe('sigmoid', () => {
  it('maps each logit independently into (0, 1)', () => {
    const p = sigmoid([0, 2, -2])
    expect(p[0]).toBeCloseTo(0.5, 6)
    expect(p[1]).toBeCloseTo(0.880797, 5)
    expect(p[2]).toBeCloseTo(0.119203, 5)
  })
  it('preserves ranking', () => {
    expect(sigmoid([-1, 3, 0.5])).toEqual([...sigmoid([-1, 3, 0.5])])
    const p = sigmoid([-1, 3, 0.5])
    expect(argmax(p)).toBe(1)
  })
})
