import { mkdir, readFile, writeFile } from "node:fs/promises"

import { cac } from "cac"

import { esc, page, stamp } from "./preview-card.ts"
import { STYLE_LINES, STYLE_SET } from "./styles.ts"

const LABELS = "labels.json"
const OUT_DIR = "preview/labels"
const STYLE_COLS = 7
const STYLE_ROWS = 3
const EMOJI_COLS = 32

const SANS = "system-ui, sans-serif"
const SERIF = "Georgia, serif"
const HAND = '"Segoe Script", cursive'

const STYLE_FACE: Record<string, { font: string; css: string }> = {
  Joyful: { font: `"Fredoka", ${SANS}`, css: "font-weight: 600" },
  Excited: { font: `"Chewy", ${SANS}`, css: "text-transform: uppercase; letter-spacing: 0.05em" },
  Hopeful: { font: `"Poppins", ${SANS}`, css: "font-weight: 500" },
  Serene: { font: `"Quicksand", ${SANS}`, css: "font-weight: 500; letter-spacing: 0.04em" },
  Tender: { font: `"Caveat", ${HAND}`, css: "font-weight: 700" },
  Playful: { font: `"Bungee", ${SANS}`, css: "" },
  Whimsical: { font: `"Griffy", ${HAND}`, css: "font-style: italic; letter-spacing: 0.02em" },
  Awed: { font: `"Luckiest Guy", ${SANS}`, css: "letter-spacing: 0.03em" },
  Earnest: { font: `"Lora", ${SERIF}`, css: "" },
  Determined: { font: `"Barlow Condensed", ${SANS}`, css: "text-transform: uppercase; font-weight: 700" },
  Proud: { font: `"Rubik", ${SANS}`, css: "text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600" },
  Wistful: { font: `"EB Garamond", ${SERIF}`, css: "font-style: italic; letter-spacing: 0.05em" },
  Melancholy: { font: `"Playfair Display", ${SERIF}`, css: "font-style: italic; font-weight: 600" },
  Anxious: { font: `"Shantell Sans", ${SANS}`, css: "font-weight: 500" },
  Tense: { font: `"Oswald", ${SANS}`, css: "text-transform: uppercase; font-weight: 500; letter-spacing: -0.01em" },
  Furious: { font: `"Anton", ${SANS}`, css: "text-transform: uppercase; letter-spacing: 0.06em" },
  Irritated: { font: `"Archivo Black", ${SANS}`, css: "text-transform: uppercase" },
  Disgusted: { font: `"Gochi Hand", ${HAND}`, css: "font-style: italic" },
  Startled: { font: `"Gaegu", ${HAND}`, css: "letter-spacing: 0.03em" },
  Sarcastic: { font: `"Bitter", ${SERIF}`, css: "font-style: italic" },
  Deadpan: { font: `"Inter", ${SANS}`, css: "" },
}

const STYLE_ACCENT: Record<string, string> = {
  Joyful: "#f5a623",
  Excited: "#f5a623",
  Hopeful: "#f5a623",
  Proud: "#f5a623",
  Serene: "#30a46c",
  Tender: "#30a46c",
  Playful: "#d6409f",
  Whimsical: "#d6409f",
  Awed: "#d6409f",
  Earnest: "#0091ff",
  Determined: "#0091ff",
  Wistful: "#5b6b8c",
  Melancholy: "#5b6b8c",
  Anxious: "#8e4ec6",
  Tense: "#8e4ec6",
  Startled: "#8e4ec6",
  Furious: "#e5484d",
  Irritated: "#e5484d",
  Disgusted: "#e5484d",
  Sarcastic: "#7a7a85",
  Deadpan: "#7a7a85",
}

type Labels = { styles: string[]; emojis: string[] }

const extraCss = (emojiRows: number) => `
* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-start;
  place-items: stretch;
  place-content: stretch;
  gap: 0.6rem;
  padding: 0.6rem;
  background: #ececee;
  font-family: system-ui, sans-serif;
}
.pl-grid {
  display: grid;
  gap: 0.35rem;
  min-height: 0;
}
#pl-styles {
  flex: 0 0 32vh;
  grid-template-columns: repeat(${STYLE_COLS}, 1fr);
  grid-template-rows: repeat(${STYLE_ROWS}, 1fr);
}
#pl-emojis {
  flex: 1 1 0;
  grid-template-columns: repeat(${EMOJI_COLS}, 1fr);
  grid-template-rows: repeat(${emojiRows}, 1fr);
}
.pl-f {
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.1em;
  padding: 0.25rem 0.6rem;
  background: #fff;
  border-left: 4px solid var(--accent, #a1a1aa);
  border-radius: 6px;
  container-type: size;
}
.pl-f-name {
  max-width: 100%;
  line-height: 1.05;
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: min(34cqh, 13cqw);
}
.pl-f-foot {
  max-width: 100%;
  text-align: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #71717a;
  font-size: max(9px, min(15cqh, 4.6cqw));
}
.pl-f-miss { color: #e5484d; }
.pl-e {
  position: relative;
  display: grid;
  place-items: center;
  background: #fff;
  border-radius: 6px;
  container-type: size;
}
.pl-e-glyph {
  line-height: 1;
  font-size: 60cqmin;
  font-family: "Noto Color Emoji", system-ui, sans-serif;
}
.pl-e-idx {
  position: absolute;
  top: 1px;
  left: 3px;
  color: #a1a1aa;
  font-family: system-ui, sans-serif;
  font-size: max(7px, 15cqmin);
}
`

function styleCell(name: string): string {
  const accent = STYLE_ACCENT[name] ?? "#a1a1aa"
  const known = STYLE_SET.has(name)
  const line = known ? STYLE_LINES[name] : "not in styles.ts"
  const face = STYLE_FACE[name] ?? { font: `"Inter", ${SANS}`, css: "" }
  const nameCss = `font-family: ${face.font}${face.css ? `; ${face.css}` : ""}`
  return (
    `<div class="pl-f" style="--accent: ${accent}">` +
    `<span class="pl-f-name" style="${esc(nameCss)}">${esc(name)}</span>` +
    `<span class="pl-f-foot${known ? "" : " pl-f-miss"}">${esc(line)}</span>` +
    `</div>`
  )
}

function emojiCell(emoji: string, i: number): string {
  return (
    `<div class="pl-e">` +
    `<span class="pl-e-idx">${i + 1}</span>` +
    `<span class="pl-e-glyph">${esc(emoji)}</span>` +
    `</div>`
  )
}

const cli = cac("preview-labels")
cli.usage("[options]")
cli.help()

if (import.meta.main) {
  cli.parse(process.argv, { run: false })
  if (cli.options.help) process.exit(0)

  const labels = JSON.parse(await readFile(LABELS, "utf8")) as Labels
  const styles = labels.styles.map(styleCell).join("\n")
  const emojis = labels.emojis.map(emojiCell).join("\n")
  const emojiRows = Math.ceil(labels.emojis.length / EMOJI_COLS)
  const body =
    `<div id="pl-styles" class="pl-grid">\n${styles}\n</div>\n` +
    `<div id="pl-emojis" class="pl-grid">\n${emojis}\n</div>`
  const html = await page({
    title: `labels — ${labels.styles.length} styles · ${labels.emojis.length} emojis`,
    extraCss: extraCss(emojiRows),
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
