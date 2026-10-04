// Explores what the real ONNX model offers for each line of showcase.txt and writes candidates.json:
// the top emojis that have an ORIGINAL Google Noto animation (byte-identical to Google's lottie.json, not one
// we generated), the top style, and the best-scoring palettes out of many samples (the model's color output is
// noisy). Nothing here touches src/showcase.json: that file is hand-picked and fixed — copy choices into it.
import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import ort from 'onnxruntime-node'
import { encode, decodeColorList, contrastRatio, toHexColor } from '../../web/src/model.js'

const here = dirname(fileURLToPath(import.meta.url))
const webPublic = join(here, '..', '..', 'web', 'public')
const meta = JSON.parse(readFileSync(join(webPublic, 'meta.json'), 'utf8'))
const noto = JSON.parse(readFileSync(join(webPublic, 'noto', 'index.json'), 'utf8'))
const char2idx = new Map([...meta.chars].map((ch, i) => [ch, i]))
const session = await ort.InferenceSession.create(join(webPublic, 'model.onnx'))
const SAMPLES = 80

const originals = new Map()
async function isOriginal(emoji) {
  const stem = noto[emoji]
  if (!stem || !existsSync(join(webPublic, 'noto', `${stem}.json`))) return false
  if (!originals.has(stem)) {
    const res = await fetch(`https://fonts.gstatic.com/s/e/notoemoji/latest/${stem}/lottie.json`)
    const local = readFileSync(join(webPublic, 'noto', `${stem}.json`), 'utf8')
    originals.set(stem, res.ok && JSON.stringify(await res.json()) === JSON.stringify(JSON.parse(local)))
  }
  return originals.get(stem)
}

// Readable text, bright and vivid background (these are feel-good cards), a calm gradient.
function score(p) {
  const contrast = Math.min(contrastRatio(p.text_color, p.bg1), contrastRatio(p.text_color, p.bg2))
  const chroma = Math.hypot(p.bg1[1], p.bg1[2])
  const light = (p.bg1[0] + p.bg2[0]) / 2
  const span = Math.hypot(p.bg1[0] - p.bg2[0], p.bg1[1] - p.bg2[1], p.bg1[2] - p.bg2[2])
  return Math.min(contrast, 7) / 7 + (1.5 * Math.min(chroma, 0.15)) / 0.15 - Math.max(0, 0.6 - light) * 6 - span * 2
}
const hue = (p) => Math.round((Math.atan2(p.bg1[2], p.bg1[1]) * 180) / Math.PI + 360) % 360

const texts = readFileSync(join(here, 'showcase.txt'), 'utf8').split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('#'))
const out = []
for (const text of texts) {
  const ids = encode(text, meta, char2idx)
  let emojis, style
  const palettes = []
  for (let i = 0; i < SAMPLES; i++) {
    const o = await session.run({ input: new ort.Tensor('int64', ids, [1, meta.max_text_len]) })
    if (i === 0) {
      const ranked = [...o.emoji_logits.data].map((v, k) => [v, k]).sort((a, b) => b[0] - a[0])
      emojis = []
      for (const [, k] of ranked) {
        if (emojis.length === 4) break
        if (await isOriginal(meta.emojis[k])) emojis.push(meta.emojis[k])
      }
      style = meta.styles[[...o.style_logits.data].reduce((m, v, k, a) => (v > a[m] ? k : m), 0)]
    }
    palettes.push(...decodeColorList(o.color.data))
  }
  const top = palettes
    .map((p) => ({ ...p, score: +score(p).toFixed(3), hue: hue(p), bg: [toHexColor(p.bg1), toHexColor(p.bg2)], fg: toHexColor(p.text_color) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 200)
  out.push({ text, emojis, style, palettes: top })
  console.log(text.padEnd(34), emojis.join(' '), style, top.slice(0, 4).map((p) => `${p.hue}°`).join(' '))
}
writeFileSync(join(here, 'candidates.json'), JSON.stringify(out, null, 2))
console.log('wrote candidates.json')
