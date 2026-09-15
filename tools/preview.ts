import { mkdir, readFile, writeFile } from "node:fs/promises"
import { cac } from "cac"
import { DATA_JSONL, LABELS_JSON } from "../files.ts"
import { CLUSTERS, DEFAULT_COLORS, FEELINGS } from "../web/src/feelings.js"
import { SCRIPT_FONT_QUERY } from "../web/src/scriptFonts.js"
import { parseJsonlText, readJsonl } from "./data/io.ts"
import { cardHtml, esc, firstEmoji, page, sample, stamp } from "./data/preview-card.ts"

const OUT_DIR = "preview"
const COLS = 5
const STYLE_SAMPLES = 2

const cli = cac("preview")
cli.usage("[file|styles|languages] [options]")
cli.option("--all", "render every emoji per card, not just the first")
cli.help()

type Row = {
  text: string
  emojis: string
  styles: string[]
  bg: [string, string]
  fg: string
}

type Labels = { styles: string[] }

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
.style-sections {
  display: flex;
  flex-direction: column;
  gap: 1.5em;
  width: 100%;
  max-width: 1400px;
}
.style-section h2 {
  margin: 0 0 0.5em;
  font: 600 1rem system-ui, sans-serif;
  color: #555;
}
.style-section .style-cards {
  display: grid;
  grid-template-columns: repeat(${STYLE_SAMPLES}, 1fr);
  gap: 1em;
}
.style-section .style-cards .card { max-width: none; }
`

function rowCard(r: Row, all: boolean, feeling: string): string {
  return cardHtml({
    text: r.text,
    emoji: all ? r.emojis : firstEmoji(r.emojis),
    feeling,
    colors: { bg1: r.bg[0], bg2: r.bg[1], text_color: r.fg },
  })
}

async function renderStyles(all: boolean): Promise<string> {
  const labels = JSON.parse(await readFile(LABELS_JSON, "utf8")) as Labels
  const rows = await readJsonl<Row>(DATA_JSONL)
  const byStyle = new Map<string, Row[]>()
  for (const style of labels.styles) byStyle.set(style, [])
  for (const row of rows) {
    for (const style of row.styles) byStyle.get(style)?.push(row)
  }
  const sections = labels.styles
    .map((style) => {
      const picks = sample(byStyle.get(style) ?? [], STYLE_SAMPLES)
      const cards = picks.map((r) => rowCard(r, all, style)).join("\n")
      return (
        `<section class="style-section">` +
        `<h2>${esc(style)} (${picks.length}/${byStyle.get(style)?.length ?? 0})</h2>` +
        `<div class="style-cards"${all ? " data-all" : ""}>\n${cards}\n</div>` +
        `</section>`
      )
    })
    .join("\n")
  const body = `<div class="style-sections">\n${sections}\n</div>`
  return page({
    title: `preview — styles — ${labels.styles.length} styles × ${STYLE_SAMPLES}${all ? " — all emojis" : ""}`,
    extraCss: EXTRA_CSS,
    body,
  })
}

const SAMPLE_LANGS: Record<string, string> = {
  latin: "en",
  cyrillic: "ru",
  greek: "el",
  arabic: "ar",
  hebrew: "he",
  devanagari: "hi",
  thai: "th",
  japanese: "ja",
  korean: "ko",
  chinese: "zh",
}

const SAMPLE_TEXT: Record<string, string> = {
  en: "this makes me so happy",
  ru: "мне так грустно сегодня",
  el: "είμαι πολύ χαρούμενος",
  ar: "أنا سعيد جدا اليوم",
  he: "אני כל כך שמח היום",
  hi: "मुझे आज बहुत खुशी है",
  th: "วันนี้ฉันมีความสุขมาก",
  ja: "今日はとても嬉しいです",
  ko: "오늘 정말 행복해요",
  zh: "我今天很开心",
}

const CLUSTER_EMOJI: Record<string, string> = {
  anger: "😠",
  joy: "😊",
  play: "🤪",
  calm: "😌",
  sad: "😢",
  anxiety: "😰",
  tender: "🥰",
  drive: "💪",
  reflective: "😑",
}

const LANG_EXTRA_CSS = `
.lang-grid {
  display: grid;
  grid-template-columns: 8em repeat(${Object.keys(SAMPLE_LANGS).length}, 1fr);
  gap: 0.75em;
  width: 100%;
}
.lang-grid .lang-col-label {
  font: 600 0.85rem system-ui, sans-serif;
  color: #555;
  align-self: center;
}
.lang-grid .card { max-width: none; aspect-ratio: 1; }
`

async function renderLanguages(): Promise<string> {
  const clusterStyle = new Map<string, string>()
  for (const cluster of Object.keys(CLUSTERS)) {
    const style = Object.entries(FEELINGS).find(([, def]) => def.cluster === cluster)?.[0]
    if (style) clusterStyle.set(cluster, style)
  }

  const scripts = Object.keys(SAMPLE_LANGS)
  const header =
    `<div class="lang-col-label"></div>` +
    scripts.map((s) => `<div class="lang-col-label">${esc(s)}</div>`).join("\n")
  const rows = [...clusterStyle.entries()]
    .map(([cluster, style]) => {
      const cards = scripts
        .map((script) => {
          const lang = SAMPLE_LANGS[script]!
          const text = SAMPLE_TEXT[lang] ?? ""
          return cardHtml({
            text,
            emoji: CLUSTER_EMOJI[cluster] ?? "🙂",
            feeling: style,
            lang,
            colors: DEFAULT_COLORS,
          })
        })
        .join("\n")
      return `<div class="lang-col-label">${esc(cluster)}<br/>(${esc(style)})</div>\n${cards}`
    })
    .join("\n")
  const body = `<div class="lang-grid">\n${header}\n${rows}\n</div>`

  const extraHead = Object.values(SCRIPT_FONT_QUERY)
    .map((q) => `<link href="https://fonts.googleapis.com/css2?${q}&display=swap" rel="stylesheet" />`)
    .join("\n")

  return page({
    title: `preview — languages — ${Object.keys(CLUSTERS).length} clusters × ${scripts.length} scripts`,
    extraCss: LANG_EXTRA_CSS,
    body,
    extraHead,
  })
}

async function renderFile(src: string | undefined, all: boolean): Promise<{ html: string; count: number }> {
  const rows = src
    ? await readJsonl<Row>(src)
    : parseJsonlText<Row>(await Bun.stdin.text(), "stdin")
  const cards = rows.map((r) => rowCard(r, all, r.styles[0] ?? "Neutral")).join("\n")
  const body = `<div class="preview-grid"${all ? " data-all" : ""}>\n${cards}\n</div>`
  const html = await page({
    title: `preview — ${src ?? "stdin"} — ${rows.length} cards${all ? " — all emojis" : ""}`,
    extraCss: EXTRA_CSS,
    body,
  })
  return { html, count: rows.length }
}

if (import.meta.main) {
  const parsed = cli.parse(process.argv, { run: false })
  if (parsed.options.help) process.exit(0)
  const SRC = parsed.args[0]
  const ALL = Boolean(parsed.options.all)

  const html =
    SRC === "styles"
      ? await renderStyles(ALL)
      : SRC === "languages"
        ? await renderLanguages()
        : (await renderFile(SRC, ALL)).html
  await mkdir(OUT_DIR, { recursive: true })
  const dest = `${OUT_DIR}/${stamp()}.html`
  await writeFile(dest, html)
  console.log(dest)
  process.exit(0)
}
