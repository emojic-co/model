// Checks every configs/*.json against the real model: each card's emoji and style must be in the model's
// top-K for its text, the emoji must be an ORIGINAL Google Noto animation, and the text must be readable.
// The outro (brand text) is only checked for animation + readability. Exits non-zero on any failure.
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import ort from 'onnxruntime-node'
import { encode, contrastRatio, CONTRAST_MIN } from '../../web/src/model.js'

const TOP_EMOJI = 5
const TOP_STYLE = 3
const MIN_CONTRAST = CONTRAST_MIN // the web app's own readability floor

const here = dirname(fileURLToPath(import.meta.url))
const pub = join(here, '..', '..', 'web', 'public')
const meta = JSON.parse(readFileSync(join(pub, 'meta.json'), 'utf8'))
const noto = JSON.parse(readFileSync(join(pub, 'noto', 'index.json'), 'utf8'))
const char2idx = new Map([...meta.chars].map((ch, i) => [ch, i]))
const session = await ort.InferenceSession.create(join(pub, 'model.onnx'))

const original = async (emoji) => {
  const stem = noto[emoji]
  const file = join(pub, 'noto', `${stem}.json`)
  if (!stem || !existsSync(file)) return false
  const res = await fetch(`https://fonts.gstatic.com/s/e/notoemoji/latest/${stem}/lottie.json`)
  return res.ok && JSON.stringify(await res.json()) === JSON.stringify(JSON.parse(readFileSync(file, 'utf8')))
}
const rank = (logits, i) => 1 + [...logits].filter((v) => v > logits[i]).length

let bad = 0
for (const f of readdirSync(join(here, 'configs')).filter((f) => f.endsWith('.json'))) {
  const config = JSON.parse(readFileSync(join(here, 'configs', f), 'utf8'))
  if (config.transitions.length !== config.cards.length) {
    console.log(`✗ ${f}: ${config.transitions.length} transitions for ${config.cards.length} cards`)
    bad++
  }
  const cards = [...config.cards.map((c) => [c, false]), [config.outro, true]]
  for (const [c, isOutro] of cards) {
    const o = await session.run({ input: new ort.Tensor('int64', encode(c.text, meta, char2idx), [1, meta.max_text_len]) })
    const er = rank(o.emoji_logits.data, meta.emojis.indexOf(c.emoji))
    const sr = rank(o.style_logits.data, meta.styles.indexOf(c.feeling))
    const cr = Math.min(contrastRatio(c.colors.text_color, c.colors.bg1), contrastRatio(c.colors.text_color, c.colors.bg2))
    const problems = []
    if (!(await original(c.emoji))) problems.push('no original Noto animation')
    if (!isOutro && er > TOP_EMOJI) problems.push(`emoji rank ${er} > ${TOP_EMOJI}`)
    if (!isOutro && sr > TOP_STYLE) problems.push(`style rank ${sr} > ${TOP_STYLE}`)
    if (cr < MIN_CONTRAST) problems.push(`contrast ${cr.toFixed(1)} < ${MIN_CONTRAST}`)
    bad += problems.length ? 1 : 0
    console.log(`${problems.length ? '✗' : '✓'} ${f.padEnd(8)} ${c.text.padEnd(30)} ${c.emoji} #${er}  ${c.feeling} #${sr}  contrast ${cr.toFixed(1)}  ${problems.join('; ')}`)
  }
}
process.exit(bad ? 1 : 0)
