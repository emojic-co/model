import { describe, it, expect } from 'vitest'
import { pickEmojiList } from './App.jsx'

const meta = { emojis: ['🍕', '🐞', '🧁'] }

describe('pickEmojiList', () => {
  const scores = {
    emoji: [0.1, 0.9, 0.4],
    fusion: [0.8, 0.2, 0.5],
    keywordRank: ['🧁', '🍕'],
  }
  it('model mode ranks by emoji score', () => {
    expect(pickEmojiList('model', scores, meta, 2).map((x) => x.emoji)).toEqual(['🐞', '🧁'])
  })
  it('fusion mode ranks by fusion score', () => {
    expect(pickEmojiList('fusion', scores, meta, 2).map((x) => x.emoji)).toEqual(['🍕', '🧁'])
  })
  it('keywords mode uses keywordRank order', () => {
    expect(pickEmojiList('keywords', scores, meta, 3).map((x) => x.emoji)).toEqual(['🧁', '🍕'])
  })
  it('returns [] before scores exist', () => {
    expect(pickEmojiList('fusion', null, meta, 5)).toEqual([])
  })
})
