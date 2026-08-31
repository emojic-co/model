import { describe, it, expect } from 'vitest'
import { cycle } from './nav'

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
