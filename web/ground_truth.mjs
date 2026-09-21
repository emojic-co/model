import * as ort from 'onnxruntime-web'
import { readFileSync } from 'fs'
import { encode, decodeColorList } from '/home/gilad/Work/emojic/web/src/model.js'

const meta = JSON.parse(readFileSync('/home/gilad/Work/emojic/web/public/meta.json', 'utf8'))
const char2idx = new Map([...meta.chars].map((ch, i) => [ch, i]))

const modelBuf = readFileSync('/home/gilad/Work/emojic/web/public/model.onnx')
const session = await ort.InferenceSession.create(modelBuf, { executionProviders: ['wasm'] })

const text = process.argv[2] || 'קפה?'
const ids = encode(text, meta, char2idx)
console.log('normalized/encoded ids (nonzero):', [...ids].filter((v) => v !== 0n))

const out = await session.run({ input: new ort.Tensor('int64', ids, [1, meta.max_text_len]) })

function sigmoid(arr) {
  return arr.map((v) => 1 / (1 + Math.exp(-v)))
}
function argmax(arr) {
  let best = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i
  return best
}

const emojiLogits = out.emoji_logits.data
const styleLogits = out.style_logits.data
const top10 = [...emojiLogits.keys()]
  .sort((a, b) => emojiLogits[b] - emojiLogits[a])
  .slice(0, 10)
  .map((i) => ({ emoji: meta.emojis[i], score: sigmoid([emojiLogits[i]])[0].toFixed(4) }))

console.log('text:', JSON.stringify(text))
console.log('top emojis:', top10)
console.log('feeling:', meta.styles[argmax(styleLogits)])
console.log('palettes:', decodeColorList(out.color.data))
