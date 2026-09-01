import { describe, it, expect } from 'vitest'
import {
  normalize,
  encode,
  argmax,
  softmax,
  decodeColors,
  decodeColorList,
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
