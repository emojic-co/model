import { describe, it, expect } from 'vitest'
import { cycle, textToHash, hashToText } from './nav'

describe('cycle', () => {
  const l = ['a', 'b', 'c']
  it('advances forward', () => expect(cycle(l, 'a', 1)).toBe('b'))
  it('advances backward', () => expect(cycle(l, 'b', -1)).toBe('a'))
  it('wraps forward past the end', () => expect(cycle(l, 'c', 1)).toBe('a'))
  it('wraps backward past the start', () => expect(cycle(l, 'a', -1)).toBe('c'))
  it('falls to the first item when current is absent, forward', () =>
    expect(cycle(l, 'z', 1)).toBe('a'))
  it('falls to the last item when current is absent, backward', () =>
    expect(cycle(l, 'z', -1)).toBe('c'))
  it('returns current for an empty list', () => expect(cycle([], 'a', 1)).toBe('a'))
})

describe('textToHash', () => {
  it('encodes text as a hash fragment', () => expect(textToHash('hello world')).toBe('#hello%20world'))
  it('trims surrounding whitespace', () => expect(textToHash('  hi  ')).toBe('#hi'))
  it('returns empty string for empty text', () => expect(textToHash('')).toBe(''))
  it('returns empty string for whitespace-only text', () => expect(textToHash('   ')).toBe(''))
  it('encodes slashes so the fragment stays intact', () =>
    expect(textToHash('a/b')).toBe('#a%2Fb'))
})

describe('hashToText', () => {
  it('decodes a hash fragment back to text', () => expect(hashToText('#hello%20world')).toBe('hello world'))
  it('returns empty string for an empty hash', () => expect(hashToText('')).toBe(''))
  it('returns empty string for a malformed hash', () => expect(hashToText('#%E0%A4%A')).toBe(''))
  it('round-trips with textToHash', () => {
    const text = 'hi there / friend'
    expect(hashToText(textToHash(text))).toBe(text)
  })
})
