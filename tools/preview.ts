import { mkdir, writeFile } from "node:fs/promises"
import { cac } from "cac"
import { parseJsonlText, readJsonl } from "./data/io.ts"
import { cardHtml, firstEmoji, page, stamp } from "./data/preview-card.ts"

const OUT_DIR = "preview"
const COLS = 5

const cli = cac("preview")
cli.usage("[file] [options]")
cli.option("--all", "render every emoji per card, not just the first")
cli.help()

type Row = {
  text: string
  emojis: string
  styles: string[]
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
  const parsed = cli.parse(process.argv, { run: false })
  if (parsed.options.help) process.exit(0)
  const SRC = parsed.args[0]
  const ALL = Boolean(parsed.options.all)

  const rows = SRC
    ? await readJsonl<Row>(SRC)
    : parseJsonlText<Row>(await Bun.stdin.text(), "stdin")
  const cards = rows
    .map((r) =>
      cardHtml({
        text: r.text,
        emoji: ALL ? r.emojis : firstEmoji(r.emojis),
        feeling: r.styles[0] ?? "Neutral",
        colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
      }),
    )
    .join("\n")
  const body = `<div class="preview-grid"${ALL ? " data-all" : ""}>\n${cards}\n</div>`
  const html = await page({
    title: `preview — ${SRC ?? "stdin"} — ${rows.length} cards${ALL ? " — all emojis" : ""}`,
    extraCss: EXTRA_CSS,
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
