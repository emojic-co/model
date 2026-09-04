import { mkdir, writeFile } from "node:fs/promises"
import { parseJsonlText, readJsonl } from "./io.ts"
import { cardHtml, firstEmoji, page, stamp } from "./preview-card.ts"

const EVAL = "eval.jsonl"
const OUT_DIR = "preview"
const COUNT = 300
const COLS = 5
const ALL = process.argv.slice(2).includes("--all")

type Row = {
  text: string
  emojis?: string
  styles?: string[]
  emoji?: string
  feeling?: string
  bg: [string, string]
  fg: string
}

const EXTRA_CSS = `
body { place-items: start center; padding: 2em 1em; }
.preview-grid {
  display: grid;
  grid-template-columns: repeat(${COLS}, 1fr);
  gap: 1em;
  width: 100%;
  max-width: 1400px;
}
.preview-grid .card { max-width: none; }
.preview-grid[data-all] .card-emoji {
  font-size: 16cqw;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 0 0.06em;
  max-width: 100%;
  transform: none;
}
`

if (import.meta.main) {
  const fromStdin = !process.stdin.isTTY
  const rows = fromStdin
    ? parseJsonlText<Row>(await Bun.stdin.text(), "stdin")
    : await readJsonl<Row>(EVAL)
  const picked = fromStdin ? rows : rows.slice(0, COUNT)
  const cards = picked
    .map((r) => {
      const emojis = r.emojis ?? r.emoji ?? ""
      return cardHtml({
        text: r.text,
        emoji: ALL ? emojis : firstEmoji(emojis),
        feeling: r.styles?.[0] ?? r.feeling ?? "Neutral",
        colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
      })
    })
    .join("\n")
  const body = `<div class="preview-grid"${ALL ? " data-all" : ""}>\n${cards}\n</div>`
  const html = await page({
    title: `preview — ${picked.length} cards${ALL ? " — all emojis" : ""}`,
    extraCss: EXTRA_CSS,
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
