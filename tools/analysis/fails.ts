import { readFile } from "node:fs/promises"
import { cac } from "cac"
import * as ort from "onnxruntime-web"
import { argmax, encode, softmax } from "../../web/src/model.js"
import { readJsonl } from "../data/io.ts"

const EVAL = "eval.jsonl"
const META = "web/public/meta.json"
const MODEL = "web/public/model.onnx"

type Row = { text: string; feeling: string; emoji: string }

type Fail = {
  text: string
  gold: string
  pred: string
  pGold: number
  pPred: number
  error: number
}

function stamp(): { header: string; file: string } {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return {
    header: `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`,
    file: `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}-${p(d.getMinutes())}`,
  }
}

function cell(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\|/g, "\\|").trim()
}

function pct(n: number, d: number): string {
  return d ? `${((100 * n) / d).toFixed(1)}%` : "n/a"
}

const cli = cac("fails")
cli.usage("[options]")
cli.help()

if (import.meta.main) {
  cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const meta = JSON.parse(await readFile(META, "utf8"))
  ort.env.wasm.numThreads = 1
  const session = await ort.InferenceSession.create(MODEL)
  const char2idx = new Map<string, number>([...meta.chars].map((c: string, i: number) => [c, i]))
  const feelIdx = new Map<string, number>(
    (meta.feelings as string[]).map((f, i) => [f, i]),
  )

  const rows = await readJsonl<Row>(EVAL)
  let scored = 0
  let skipped = 0
  let correct = 0
  const fails: Fail[] = []

  for (const r of rows) {
    const gold = feelIdx.get(r.feeling)
    if (gold === undefined) {
      skipped++
      continue
    }
    const ids = encode(r.text, meta, char2idx)
    const tensor = new ort.Tensor("int64", ids, [1, meta.max_text_len])
    const out = await session.run({ input: tensor })
    const probs = softmax(Array.from(out.feeling_logits.data as Float32Array))
    const pred = argmax(probs)
    scored++
    if (pred === gold) {
      correct++
      continue
    }
    fails.push({
      text: r.text,
      gold: r.feeling,
      pred: meta.feelings[pred],
      pGold: probs[gold],
      pPred: probs[pred],
      error: 1 - probs[gold],
    })
  }

  fails.sort((a, b) => b.error - a.error)

  const { header } = stamp()
  const doc = [
    `# feeling fails — ${header}`,
    "",
    `model \`${MODEL}\` · ${rows.length} eval rows · ${scored} scored · ${skipped} skipped (feeling not in \`meta.json\`)`,
    "",
    `feeling top-1 accuracy **${pct(correct, scored)}** · **${fails.length}** fails`,
    "",
    "Ordered by softmax miss probability `error = 1 - p(gold)`, worst first.",
    "",
    "| # | error | p(gold) | p(pred) | text | gold | predicted |",
    "| ---: | ---: | ---: | ---: | --- | --- | --- |",
  ]
  fails.forEach((f, i) => {
    doc.push(
      `| ${i + 1} | ${f.error.toFixed(3)} | ${f.pGold.toFixed(3)} | ${f.pPred.toFixed(3)} | ${cell(f.text)} | ${cell(f.gold)} | ${cell(f.pred)} |`,
    )
  })
  doc.push("")

  console.log(doc.join("\n"))
  process.exit(0)
}
