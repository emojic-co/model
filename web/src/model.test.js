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
  hexToOklab,
  toCssOklab,
  contrastRatio,
  fixContrast,
  mixColors,
  patternTint,
  CONTRAST_MIN,
  BLACK,
  WHITE,
} from './model'

const CHARS = '·abcdefghijklmnopqrstuvwxyz0123456789!?:()@$%&* '
const idx = new Map([...CHARS].map((c, i) => [c, i]))

describe('normalize', () => {
  it('lower-cases text and collapses whitespace', () => {
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
  it('maps unit -1/1 to black/white OKLab triples', () => {
    const decoded = decodeColors([1, 0, 0, -1, 0, 0, 1, 0, 0])
    expect(decoded.bg1).toEqual([1, 0, 0])
    expect(decoded.bg2).toEqual([0, 0, 0])
    expect(decoded.text_color).toEqual([1, 0, 0])
  })
  it('maps unit 0 to the neutral midpoint L=0.5, a=0, b=0', () => {
    const decoded = decodeColors([0, 0, 0, 0, 0, 0, 0, 0, 0])
    expect(decoded.bg1).toEqual([0.5, 0, 0])
  })
  it('scales a/b by the given ab_range', () => {
    const decoded = decodeColors([0, 1, -1, 0, 0, 0, 0, 0, 0], 0.3)
    expect(decoded.bg1[0]).toBeCloseTo(0.5, 10)
    expect(decoded.bg1[1]).toBeCloseTo(0.3, 10)
    expect(decoded.bg1[2]).toBeCloseTo(-0.3, 10)
  })
})

describe('decodeColorList', () => {
  it('chunks a flat 45-value buffer into 5 palettes', () => {
    const list = decodeColorList(new Float32Array(45).fill(0))
    expect(list).toHaveLength(5)
    for (const p of list) {
      expect(p.bg1).toEqual([0.5, 0, 0])
      expect(p.bg2).toEqual([0.5, 0, 0])
      expect(p.text_color).toEqual([0.5, 0, 0])
    }
  })
  it('decodes each chunk independently', () => {
    const swatch = (n) => [n, 0, 0]
    const flat = [
      ...swatch(-1), ...swatch(-1), ...swatch(-1),
      ...swatch(1), ...swatch(1), ...swatch(1),
    ]
    expect(decodeColorList(flat)).toEqual([
      { bg1: [0, 0, 0], bg2: [0, 0, 0], text_color: [0, 0, 0] },
      { bg1: [1, 0, 0], bg2: [1, 0, 0], text_color: [1, 0, 0] },
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

describe('hexToOklab / toCssOklab', () => {
  it('converts white and black hex to their exact OKLab triples', () => {
    expect(hexToOklab('#ffffff')[0]).toBeCloseTo(1, 5)
    expect(hexToOklab('#000000')).toEqual([0, 0, 0])
  })
  it('formats an OKLab triple as a CSS oklab() string', () => {
    expect(toCssOklab([0.6, -0.05, 0.12])).toBe('oklab(60.00% -0.0500 0.1200)')
  })
})

describe('contrastRatio', () => {
  it('is 21 for black on white and 1 for a colour on itself', () => {
    expect(contrastRatio(BLACK, WHITE)).toBeCloseTo(21, 4)
    const c = hexToOklab('#3a7bd5')
    expect(contrastRatio(c, c)).toBeCloseTo(1, 5)
  })
  it('is symmetric', () => {
    const a = hexToOklab('#123456')
    const b = hexToOklab('#abcdef')
    expect(contrastRatio(a, b)).toBeCloseTo(contrastRatio(b, a), 10)
  })
})

describe('fixContrast', () => {
  it('leaves a readable palette untouched (same object)', () => {
    const p = {
      bg1: hexToOklab('#a8e2f4'),
      bg2: hexToOklab('#78c9f4'),
      text_color: hexToOklab('#282e36'),
    }
    expect(fixContrast(p)).toBe(p)
  })

  it('repairs a low-contrast palette so both stops clear the threshold', () => {
    const p = {
      bg1: hexToOklab('#2b2b2b'),
      bg2: hexToOklab('#3a3a3a'),
      text_color: hexToOklab('#444444'),
    }
    const fixed = fixContrast(p)
    expect(fixed.text_color).not.toBe(p.text_color)
    expect(fixed.bg1).toBe(p.bg1)
    expect(fixed.bg2).toBe(p.bg2)
    expect(contrastRatio(fixed.text_color, fixed.bg1)).toBeGreaterThanOrEqual(CONTRAST_MIN)
    expect(contrastRatio(fixed.text_color, fixed.bg2)).toBeGreaterThanOrEqual(CONTRAST_MIN)
  })

  it('nudges lightness while roughly preserving hue', () => {
    const p = {
      bg1: hexToOklab('#c94f4f'),
      bg2: hexToOklab('#d46b4b'),
      text_color: hexToOklab('#b84a3a'),
    }
    const fixed = fixContrast(p)
    const [, a0, b0] = p.text_color
    const [, a1, b1] = fixed.text_color
    expect(Math.sign(a1)).toBe(Math.sign(a0))
    expect(Math.sign(b1)).toBe(Math.sign(b0))
  })

  it('respects a custom threshold', () => {
    const p = {
      bg1: hexToOklab('#ffffff'),
      bg2: hexToOklab('#f4f4f4'),
      text_color: hexToOklab('#8a8a8a'),
    }
    const fixed = fixContrast(p, 4.5)
    expect(minOf(fixed)).toBeGreaterThanOrEqual(4.5)
  })
})

function minOf({ bg1, bg2, text_color }) {
  return Math.min(contrastRatio(text_color, bg1), contrastRatio(text_color, bg2))
}

describe('mixColors', () => {
  it('at t=0 returns the first color, at t=1 the second', () => {
    const a = hexToOklab('#ff0000')
    const b = hexToOklab('#0000ff')
    expect(mixColors(a, b, 0)).toEqual(a)
    expect(mixColors(a, b, 1)).toEqual(b)
  })
  it('at t=0.5 is the midpoint on each channel', () => {
    expect(mixColors([0, 0, 0], [1, 0.4, -0.4], 0.5)).toEqual([0.5, 0.2, -0.2])
  })
})

describe('patternTint', () => {
  it('floors lightness at 0.94 while keeping hue', () => {
    const tint = patternTint([0.5, 0.1, -0.1], [0.5, 0.1, -0.1])
    expect(tint).toEqual([0.94, 0.1, -0.1])
  })
  it('keeps lightness above 0.94 if the mix is already lighter', () => {
    const tint = patternTint([0.98, 0.05, 0], [0.98, 0.05, 0])
    expect(tint[0]).toBeCloseTo(0.98, 10)
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
