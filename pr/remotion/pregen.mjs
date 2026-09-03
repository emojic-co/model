import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import ort from 'onnxruntime-node'
import { normalize, encode, decodeColorList, sigmoid, argmax } from '../../web/src/model.js'

const here = dirname(fileURLToPath(import.meta.url))
const webPublic = join(here, '..', '..', 'web', 'public')

const meta = JSON.parse(readFileSync(join(webPublic, 'meta.json'), 'utf8'))
const char2idx = new Map([...meta.chars].map((ch, i) => [ch, i]))

const slug = (t) =>
  t
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')

const texts = readFileSync(join(here, 'texts.txt'), 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter(Boolean)

const session = await ort.InferenceSession.create(join(webPublic, 'model.onnx'))

async function predict(text) {
  const ids = encode(text, meta, char2idx)
  const tensor = new ort.Tensor('int64', ids, [1, meta.max_text_len])
  const out = await session.run({ input: tensor })
  const emoji = sigmoid(out.emoji_logits.data)
  const feeling = sigmoid(out.style_logits.data)
  const palette = decodeColorList(out.color.data)[0]
  return {
    emoji: meta.emojis[argmax(emoji)],
    feeling: meta.styles[argmax(feeling)],
    ...palette,
  }
}

const data = {}
for (const text of texts) {
  const s = slug(text)
  const frames = []
  for (let k = 1; k <= text.length; k++) {
    const prefix = text.slice(0, k)
    const normLen = normalize(prefix, char2idx).length
    const p = await predict(prefix)
    frames.push({ k, normLen, meaningful: normLen >= 3, ...p })
  }
  data[s] = { text, slug: s, frames }
  mkdirSync(join(here, 'data'), { recursive: true })
  writeFileSync(join(here, 'data', `${s}.json`), JSON.stringify(data[s], null, 2))
  const last = frames[frames.length - 1]
  console.log(`${s.padEnd(34)} ${last.emoji}  ${last.feeling}`)
}

mkdirSync(join(here, 'src'), { recursive: true })
writeFileSync(join(here, 'src', 'data.json'), JSON.stringify(data, null, 2))
console.log(`\nwrote src/data.json (${Object.keys(data).length} texts)`)
