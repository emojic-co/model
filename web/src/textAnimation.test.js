import { describe, it, expect } from 'vitest'
import { TEXT_ANIMATIONS as A, splitWords, scheduleFor, keyframesFor, hash } from './textAnimation'

describe('splitWords', () => {
  it('splits words into graphemes, keeping ZWJ emoji and Hebrew intact', () => {
    expect(splitWords('hi 👨‍👩‍👧')).toEqual([['h', 'i'], ['👨‍👩‍👧']])
    expect(splitWords('שלום עולם')[0]).toHaveLength(4)
  })
})

describe('scheduleFor', () => {
  it('caps total duration for long text', () => {
    const { motif, delays } = scheduleFor(A, 'rise', 'Hopeful', 200)
    expect(Math.max(...delays) + motif.durationMs).toBeLessThanOrEqual(A.timing.maxTotalMs + 1)
  })
  it('applies orders and style overrides', () => {
    expect(scheduleFor(A, 'drop', 'Wistful', 3).delays[2]).toBe(0)
    const c = scheduleFor(A, 'bloom', 'Tender', 5).delays
    expect(c[2]).toBe(0)
    expect(c[0]).toBe(c[4])
    expect(scheduleFor(A, 'settle', 'Serene', 2).motif.staggerMs).toBe(110)
  })
  it('uses a stable hash', () => {
    expect(hash(0)).toBe(hash(0))
    expect(hash(3)).toBeGreaterThanOrEqual(0)
    expect(hash(3)).toBeLessThan(1)
  })
})

describe('keyframesFor', () => {
  it('carries fields forward and mirrors on odd units for alternate motifs', () => {
    const m = A.motifs.spin
    expect(keyframesFor(m, 0)[0].transform).toContain('rotate(-140deg)')
    expect(keyframesFor(m, 1)[0].transform).toContain('rotate(140deg)')
    expect(keyframesFor(A.motifs.slam, 0).at(-1).opacity).toBe(1)
  })
})
