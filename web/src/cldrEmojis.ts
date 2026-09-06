import { Index } from 'flexsearch'
import cldrEmojiIndex from './data/cldr-emoji-index.json'

const LIMIT = 50

let index: Index | null = null

function getIndex(): Index {
  if (index) return index
  index = new Index({ tokenize: 'strict' })
  cldrEmojiIndex.forEach((entry, id) => index!.add(id, entry.text))
  return index
}

export function cldrEmojis(text: string): string[] {
  const ids = getIndex().search(text, { limit: LIMIT, suggest: true }) as number[]
  return ids.map((id) => cldrEmojiIndex[id].emoji)
}
