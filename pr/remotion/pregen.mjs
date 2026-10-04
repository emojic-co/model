// Runs the real ONNX model on each line of showcase.txt and writes src/showcase.json:
// `text | palette | emoji | feeling` (all but text optional). emoji = highest-logit emoji that has an animated Lottie clone, style = argmax, palette = `| n` (default 0).
// Also copies the needed Noto Lottie clips into public/noto/ for the composition.
import { readFileSync, writeFileSync, mkdirSync, copyFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import ort from 'onnxruntime-node'
import { encode, decodeColorList, argmax } from '../../web/src/model.js'

const here = dirname(fileURLToPath(import.meta.url))
const webPublic = join(here, '..', '..', 'web', 'public')
const meta = JSON.parse(readFileSync(join(webPublic, 'meta.json'), 'utf8'))
const noto = JSON.parse(readFileSync(join(webPublic, 'noto', 'index.json'), 'utf8'))
const char2idx = new Map([...meta.chars].map((ch, i) => [ch, i]))
const session = await ort.InferenceSession.create(join(webPublic, 'model.onnx'))

const lines = readFileSync(join(here, 'showcase.txt'), 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l && !l.startsWith('#'))

const cards = []
mkdirSync(join(here, 'public', 'noto'), { recursive: true })
for (const line of lines) {
  const [text, pal = '0', forceEmoji, forceFeeling] = line.split('|').map((s) => s.trim())
  const ids = encode(text, meta, char2idx)
  const out = await session.run({ input: new ort.Tensor('int64', ids, [1, meta.max_text_len]) })
  const ranked = [...out.emoji_logits.data].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])
  const emoji = forceEmoji || meta.emojis[(ranked.find(([, i]) => noto[meta.emojis[i]]) ?? ranked[0])[1]]
  const palette = decodeColorList(out.color.data)[Number(pal)]
  const card = { text, emoji, feeling: forceFeeling || meta.styles[argmax(out.style_logits.data)], ...palette }
  cards.push(card)
  if (noto[emoji]) copyFileSync(join(webPublic, 'noto', `${noto[emoji]}.json`), join(here, 'public', 'noto', `${noto[emoji]}.json`))
  console.log(text.padEnd(34), card.emoji, card.feeling, noto[emoji] ? 'lottie' : 'static')
}
const used = Object.fromEntries(cards.filter((c) => noto[c.emoji]).map((c) => [c.emoji, noto[c.emoji]]))
writeFileSync(join(here, 'public', 'noto', 'index.json'), JSON.stringify(used))
writeFileSync(join(here, 'src', 'showcase.json'), JSON.stringify(cards, null, 2))
console.log(`wrote src/showcase.json (${cards.length} cards)`)
