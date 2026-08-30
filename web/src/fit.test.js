import { describe, it, expect } from 'vitest'
import { wrapLines, fitCanvasFont } from './fit'

const widthAt = (s, px) => s.length * px * 0.5

describe('wrapLines', () => {
  it('wraps on width and caps at maxLines', () => {
    const lines = wrapLines((s) => s.length * 10, 'aa bb cc dd', 50, 2)
    expect(lines).toEqual(['aa bb', 'cc dd'])
  })
  it('keeps a single short line', () => {
    expect(wrapLines((s) => s.length, 'hi there', 100, 4)).toEqual(['hi there'])
  })
})

describe('fitCanvasFont', () => {
  it('returns max when the text easily fits', () => {
    const px = fitCanvasFont({
      text: 'hi', maxWidth: 500, maxHeight: 500,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBe(100)
  })
  it('shrinks long text below max', () => {
    const long = 'x'.repeat(48)
    const px = fitCanvasFont({
      text: long, maxWidth: 400, maxHeight: 260,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBeGreaterThanOrEqual(20)
    expect(px).toBeLessThan(100)
  })
  it('never returns below min', () => {
    const px = fitCanvasFont({
      text: 'y'.repeat(500), maxWidth: 50, maxHeight: 50,
      min: 20, max: 100, lineHeight: 1.3, widthAt,
    })
    expect(px).toBe(20)
  })
})
