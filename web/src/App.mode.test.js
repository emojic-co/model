import { describe, it, expect } from 'vitest'
import { pickEmojiList } from './App.jsx'

const meta = { emojis: ['🍕', '🐞', '🧁'] }

describe('pickEmojiList', () => {
  const scores = { emoji: [0.1, 0.9, 0.4] }
  it('ranks by emoji score', () => {
    expect(pickEmojiList(scores, meta, 2).map((x) => x.emoji)).toEqual(['🐞', '🧁'])
  })
  it('returns [] before scores exist', () => {
    expect(pickEmojiList(null, meta, 5)).toEqual([])
  })
})
