import { mkdir, readFile, writeFile } from "node:fs/promises"
import { cac } from "cac"
import * as ort from "onnxruntime-web"
import { argmax, decodeColorList, encode } from "../../web/src/model.js"
import { readJsonl } from "./io.ts"
import { cardHtml, esc, page, sample, stamp } from "./preview-card.ts"

const EVAL = "eval.jsonl"
const META = "web/public/meta.json"
const MODEL = "web/public/model.onnx"
const OUT_DIR = "preview/model"
const COUNT = 20

type Row = { text: string; feeling: string; emoji: string; bg: [string, string]; fg: string }

const EXTRA_CSS = `
body { place-items: start center; padding: 2em 1em; }
.preview-model {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75em 1em;
  width: 100%;
  max-width: 900px;
}
.preview-model .card { max-width: none; }
.pm-head {
  font-size: 0.75em;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  text-align: center;
}
.pm-cap {
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.5em 1.5em;
  font-size: 0.8em;
  color: var(--muted);
  padding-bottom: 0.9em;
  border-bottom: 1px solid var(--line);
}
.pm-cap b { color: var(--ink); font-weight: 600; }
.pm-ok { color: #2e7d32; }
.pm-no { color: #c0392b; }
`

const cli = cac("preview-model")
cli.usage("[options]")
cli.help()

if (import.meta.main) {
  cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const meta = JSON.parse(await readFile(META, "utf8"))
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(MODEL)
  const char2idx = new Map<string, number>([...meta.chars].map((c: string, i: number) => [c, i]))

  const rows = await readJsonl<Row>(EVAL)
  const picked = sample(rows, COUNT)

  const cells: string[] = [
    `<div class="pm-head">ground truth</div>`,
    `<div class="pm-head">model</div>`,
  ]
  for (const r of picked) {
    const ids = encode(r.text, meta, char2idx)
    const tensor = new ort.Tensor("int64", ids, [1, meta.max_text_len])
    const out = await session.run({ input: tensor })
    const pf: string = meta.feelings[argmax(Array.from(out.feeling_logits.data as Float32Array))]
    const pe: string = meta.emojis[argmax(Array.from(out.emoji_logits.data as Float32Array))]
    const pc = decodeColorList(out.color.data as Float32Array)[0]
    const truth = cardHtml({
      text: r.text,
      emoji: r.emoji,
      feeling: r.feeling,
      colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
    })
    const model = cardHtml({ text: r.text, emoji: pe, feeling: pf, colors: pc })
    const fMark =
      pf === r.feeling ? `<span class="pm-ok">✓</span>` : `<span class="pm-no">✗</span>`
    const eMark =
      pe === r.emoji ? `<span class="pm-ok">✓</span>` : `<span class="pm-no">✗</span>`
    const cap =
      `<div class="pm-cap">` +
      `<span>truth <b>${esc(r.feeling)}</b> ${esc(r.emoji)}</span>` +
      `<span>model <b>${esc(pf)}</b> ${esc(pe)}</span>` +
      `<span>feeling ${fMark} · emoji ${eMark}</span>` +
      `</div>`
    cells.push(truth, model, cap)
  }

  const body = `<div class="preview-model">\n${cells.join("\n")}\n</div>`
  const html = await page({
    title: `preview-model — ${picked.length} pairs`,
    extraCss: EXTRA_CSS,
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
