import { mkdir, writeFile } from "node:fs/promises"
import { readJsonl } from "./io.ts"
import { cardHtml, page, stamp } from "./preview-card.ts"

const SRC = process.argv[2] ?? "pred.jsonl"
const OUT_DIR = "report/preview"
const COLS = 5

type Row = { text: string; feeling: string; emoji: string; bg: [string, string]; fg: string }

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
`

if (import.meta.main) {
  const rows = await readJsonl<Row>(SRC)
  const cards = rows
    .map((r) =>
      cardHtml({
        text: r.text,
        emoji: r.emoji,
        feeling: r.feeling,
        colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
      }),
    )
    .join("\n")
  const body = `<div class="preview-grid">\n${cards}\n</div>`
  const html = await page({
    title: `preview — ${SRC} — ${rows.length} cards`,
    extraCss: EXTRA_CSS,
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/pred-${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
