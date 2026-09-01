import { readFile } from "node:fs/promises"
import { resolveFeeling } from "../../web/src/feelings.js"

export type Colors = { bg1: string; bg2: string; text_color: string }
export type CardData = { text: string; emoji: string; feeling: string; colors: Colors }

export function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
}

export function styleToCss(obj: Record<string, string | number> | undefined): string {
  if (!obj) return ""
  return Object.entries(obj)
    .map(([k, v]) => {
      const prop = k.startsWith("--") ? k : k.replace(/[A-Z]/g, (m) => "-" + m.toLowerCase())
      return `${prop}: ${v}`
    })
    .join("; ")
}

export function cardHtml({ text, emoji, feeling, colors }: CardData): string {
  const r = resolveFeeling(feeling)
  const placeholder = !text.trim()
  const displayText = placeholder ? "What's on your mind?" : text
  const cardStyle = styleToCss({
    backgroundImage: `linear-gradient(135deg, ${colors.bg1}, ${colors.bg2})`,
    color: colors.text_color,
    fontFamily: r.font,
    ...r.vars,
  })
  const textStyle = styleToCss(r.style)
  return (
    `<div class="card" data-feeling="${esc(feeling)}" data-cluster="${esc(r.cluster)}"` +
    ` data-entrance="${esc(r.entrance)}" data-emoji="${esc(r.emoji)}" data-phase="in"` +
    ` style="${esc(cardStyle)}">` +
    `<span class="card-emoji">${esc(emoji)}</span>` +
    `<div class="card-text-box">` +
    `<p class="card-text${placeholder ? " card-text-placeholder" : ""}"` +
    `${textStyle ? ` style="${esc(textStyle)}"` : ""}>${esc(displayText)}</p>` +
    `</div></div>`
  )
}

const FIT_SCRIPT = `
function fitText(el, min, max) {
  var fits = function () {
    return el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight
  }
  var lo = min
  var hi = max
  el.style.fontSize = lo + 'cqw'
  if (!fits()) return
  for (var i = 0; i < 22; i++) {
    var mid = (lo + hi) / 2
    el.style.fontSize = mid + 'cqw'
    if (fits()) lo = mid
    else hi = mid
  }
  el.style.fontSize = lo + 'cqw'
}
function fitAll() {
  document.querySelectorAll('.card-text-box').forEach(function (el) {
    fitText(el, 5, 13)
  })
}
fitAll()
if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitAll)
window.addEventListener('resize', fitAll)
`

export function stamp(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, "0")
  return `${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}:${p(d.getMinutes())}`
}

export function sample<T>(rows: T[], n: number): T[] {
  const a = [...rows]
  const k = Math.min(n, a.length)
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(Math.random() * (a.length - i))
    ;[a[i], a[j]] = [a[j], a[i]]
  }
  return a.slice(0, k)
}

export async function page(opts: {
  title: string
  extraCss: string
  body: string
}): Promise<string> {
  const stylesCss = await readFile("web/src/styles.css", "utf8")
  const indexHtml = await readFile("web/index.html", "utf8")
  const m = indexHtml.match(/href="(https:\/\/fonts\.googleapis\.com\/[^"]+)"/)
  const fontsLink = m ? `<link href="${m[1]}" rel="stylesheet" />` : ""
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${esc(opts.title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
${fontsLink}
<style>
${stylesCss}
${opts.extraCss}
</style>
</head>
<body>
${opts.body}
<script>
${FIT_SCRIPT}
</script>
</body>
</html>
`
}
