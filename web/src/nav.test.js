import { describe, it, expect } from 'vitest'
import { cycle, textToPath, pathToText } from './nav'

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

describe('textToPath', () => {
  it('encodes text as a path segment', () => expect(textToPath('hello world')).toBe('/hello%20world'))
  it('trims surrounding whitespace', () => expect(textToPath('  hi  ')).toBe('/hi'))
  it('returns root for empty text', () => expect(textToPath('')).toBe('/'))
  it('returns root for whitespace-only text', () => expect(textToPath('   ')).toBe('/'))
  it('encodes slashes so path segments stay intact', () =>
    expect(textToPath('a/b')).toBe('/a%2Fb'))
})

describe('pathToText', () => {
  it('decodes a path segment back to text', () => expect(pathToText('/hello%20world')).toBe('hello world'))
  it('returns empty string for root', () => expect(pathToText('/')).toBe(''))
  it('returns empty string for a malformed path', () => expect(pathToText('/%E0%A4%A')).toBe(''))
  it('round-trips with textToPath', () => {
    const text = 'hi there / friend'
    expect(pathToText(textToPath(text))).toBe(text)
  })
})
