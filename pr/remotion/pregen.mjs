import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import ort from 'onnxruntime-node'
import { encode, decodeColorList, sigmoid, argmax } from '../../web/src/model.js'

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
  const palette = decodeColorList(out.color.data)[0]
  return {
    emoji: meta.emojis[argmax(sigmoid(out.emoji_logits.data))],
    feeling: meta.styles[argmax(sigmoid(out.style_logits.data))],
    ...palette,
  }
}

const data = {}
mkdirSync(join(here, 'data'), { recursive: true })
for (const text of texts) {
  const s = slug(text)
  data[s] = { text, slug: s, ...(await predict(text)) }
  writeFileSync(join(here, 'data', `${s}.json`), JSON.stringify(data[s], null, 2))
  console.log(`${s.padEnd(34)} ${data[s].emoji}  ${data[s].feeling}`)
}

mkdirSync(join(here, 'src'), { recursive: true })
writeFileSync(join(here, 'src', 'data.json'), JSON.stringify(data, null, 2))
console.log(`\nwrote src/data.json (${Object.keys(data).length} texts)`)
