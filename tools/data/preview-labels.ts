import { mkdir, readFile, writeFile } from "node:fs/promises"
import { FEELINGS, resolveFeeling } from "../../web/src/feelings.js"
import { esc, page, stamp, styleToCss } from "./preview-card.ts"

const LABELS = "labels.json"
const OUT_DIR = "report/preview-labels"

const CLUSTER_ACCENT: Record<string, string> = {
  anger: "#e5484d",
  joy: "#f5a623",
  play: "#d6409f",
  calm: "#30a46c",
  sad: "#5b6b8c",
  anxiety: "#8e4ec6",
  tender: "#e93d82",
  drive: "#0091ff",
  reflective: "#7a7a85",
}

type Labels = { feelings: string[]; emojis: string[] }

const EXTRA_CSS = `
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
  flex: 1 1 0;
  min-height: 0;
}
.pl-f {
  position: relative;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.12em;
  padding: 0.25rem 0.5rem;
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
  font-size: min(34cqh, 11cqw);
}
.pl-f-foot {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #71717a;
  font-size: max(9px, min(15cqh, 6cqw));
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
  font-size: 56cqmin;
  font-family: "Noto Color Emoji", system-ui, sans-serif;
}
.pl-e-idx {
  position: absolute;
  top: 2px;
  left: 4px;
  color: #a1a1aa;
  font-family: system-ui, sans-serif;
  font-size: max(8px, 13cqmin);
}
`

function fontName(font: string): string {
  const m = font.match(/"([^"]+)"/)
  return m ? m[1] : font.split(",")[0].trim()
}

function feelingCell(name: string): string {
  const r = resolveFeeling(name)
  const accent = CLUSTER_ACCENT[r.cluster] ?? "#a1a1aa"
  const nameStyle = styleToCss({ fontFamily: r.font, ...r.style })
  const missing = !(name in FEELINGS)
  const foot = `${esc(fontName(r.font))} · ${esc(r.cluster)}${missing ? " · missing" : ""}`
  return (
    `<div class="pl-f" style="--accent: ${accent}">` +
    `<span class="pl-f-name" style="${esc(nameStyle)}">${esc(name)}</span>` +
    `<span class="pl-f-foot${missing ? " pl-f-miss" : ""}">${foot}</span>` +
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

function layoutScript(nf: number, ne: number): string {
  return `
(function () {
  var NF = ${nf}, NE = ${ne}, TF = 2.8, TE = 1.0
  var fg = document.getElementById('pl-feelings')
  var eg = document.getElementById('pl-emojis')
  function layout() {
    var rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
    var pad = 0.6 * rem, gap = 0.6 * rem
    var innerW = document.documentElement.clientWidth - 2 * pad
    var innerH = document.documentElement.clientHeight - 2 * pad - gap
    var best = null
    for (var cf = 2; cf <= Math.min(6, NF); cf++) {
      var rf = Math.ceil(NF / cf)
      var idF = rf * (innerW / cf) / TF
      for (var ce = 6; ce <= Math.min(32, NE); ce++) {
        var re = Math.ceil(NE / ce)
        var idE = re * (innerW / ce) / TE
        var err = Math.abs(idF + idE - innerH)
        if (!best || err < best.err) best = { cf: cf, rf: rf, ce: ce, re: re, idF: idF, idE: idE, err: err }
      }
    }
    fg.style.gridTemplateColumns = 'repeat(' + best.cf + ', 1fr)'
    fg.style.gridTemplateRows = 'repeat(' + best.rf + ', 1fr)'
    fg.style.flexGrow = String(best.idF)
    eg.style.gridTemplateColumns = 'repeat(' + best.ce + ', 1fr)'
    eg.style.gridTemplateRows = 'repeat(' + best.re + ', 1fr)'
    eg.style.flexGrow = String(best.idE)
  }
  layout()
  addEventListener('resize', layout)
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(layout)
})()
`
}

if (import.meta.main) {
  const labels = JSON.parse(await readFile(LABELS, "utf8")) as Labels
  const feelings = labels.feelings.map(feelingCell).join("\n")
  const emojis = labels.emojis.map(emojiCell).join("\n")
  const body =
    `<div id="pl-feelings" class="pl-grid">\n${feelings}\n</div>\n` +
    `<div id="pl-emojis" class="pl-grid">\n${emojis}\n</div>\n` +
    `<script>${layoutScript(labels.feelings.length, labels.emojis.length)}</script>`
  const html = await page({
    title: `labels — ${labels.feelings.length} feelings · ${labels.emojis.length} emojis`,
    extraCss: EXTRA_CSS,
    body,
  })
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
