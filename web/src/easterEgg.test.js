import { describe, expect, it } from 'vitest'
import { eggPieces, isEasterEgg, EGG_COUNT } from './easterEgg'

describe('easter egg', () => {
  it('matches the phrase ignoring case and surrounding space', () => {
    expect(isEasterEgg('emojify.ing')).toBe(true)
    expect(isEasterEgg('  Emojify.ING ')).toBe(true)
    expect(isEasterEgg('emojify')).toBe(false)
  })
  it('builds a fixed number of pieces', () => {
    expect(eggPieces()).toHaveLength(EGG_COUNT)
  })
})
