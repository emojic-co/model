import { describe, expect, it } from 'vitest'
import { playerKey, shapeKey } from './useCardPlayer'

const base = { emoji: '😀', feeling: 'Joyful', lang: 'en', text: 'hi', colors: { bg1: [0.5, 0, 0], bg2: [0.6, 0, 0], text_color: [0, 0, 0] } }

describe('playerKey', () => {
  it('changes for emoji, feeling, lang, text and colors', () => {
    const k = playerKey(base)
    expect(playerKey({ ...base, emoji: '😢' })).not.toBe(k)
    expect(playerKey({ ...base, feeling: 'Serene' })).not.toBe(k)
    expect(playerKey({ ...base, lang: 'he' })).not.toBe(k)
    expect(playerKey({ ...base, text: 'hi!' })).not.toBe(k)
    expect(playerKey({ ...base, colors: { ...base.colors, bg1: [0.4, 0, 0] } })).not.toBe(k)
  })
  it('is stable for equal content', () => {
    expect(playerKey({ ...base, colors: { ...base.colors } })).toBe(playerKey(base))
  })
})

describe('shapeKey', () => {
  it('ignores text and colors (those keep the old frame until the new one is ready)', () => {
    const k = shapeKey(base)
    expect(shapeKey({ ...base, text: 'other', colors: { ...base.colors, bg1: [0.1, 0, 0] } })).toBe(k)
    expect(shapeKey({ ...base, emoji: '😢' })).not.toBe(k)
    expect(shapeKey({ ...base, feeling: 'Serene' })).not.toBe(k)
    expect(shapeKey({ ...base, lang: 'he' })).not.toBe(k)
  })
})
