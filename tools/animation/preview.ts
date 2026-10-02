// Writes animation/preview.html: every frame of an emoji at once (static original first, with its description),
// plus a live loop. With no emoji, one section per emoji that has frames.
//   bun run animation-preview 🔴 [--open]
//   bun run animation-preview [--open]
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs"

import { cac } from "cac"

import { ANIMATION_DIR } from "../../files.ts"
import { FRAMES, framePath, jsonPath, parseEmoji, stemOf } from "./common.ts"

const OUT = `${ANIMATION_DIR}/preview.html`
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;")
const rel = (p: string) => p.slice(ANIMATION_DIR.length + 1)

function section(stem: string): string {
  const desc: string[] = existsSync(jsonPath(stem)) ? JSON.parse(readFileSync(jsonPath(stem), "utf8")) : []
  const frames = Array.from({ length: FRAMES }, (_, i) => i + 1)
  const cells = frames.map((n) => {
    const p = framePath(stem, n)
    const img = existsSync(p) ? `<img src="${rel(p)}" alt="">` : `<div class="missing">missing</div>`
    return `<figure>${img}<figcaption><b>${n}</b> ${esc(desc[n - 1] ?? "")}</figcaption></figure>`
  })
  const code = [...stem.split("_")].map((h) => String.fromCodePoint(parseInt(h, 16))).join("")
  return `<section data-stem="${stem}"><h2>${esc(code)} <small>${stem}</small></h2>
<div class="row"><div class="loop"><img class="live" src="${rel(framePath(stem, 1))}" alt=""><label>fps <input type="number" class="fps" value="12" min="1" max="60"></label></div>
<div class="grid">${cells.join("")}</div></div></section>`
}

const cli = cac("animation-preview")
cli
  .command("[emoji]", "preview all frames at once")
  .option("--open", "open the page in the browser")
  .action((arg: string | undefined, o: { open?: boolean }) => {
    const stems = arg
      ? [stemOf(parseEmoji(arg))]
      : existsSync(`${ANIMATION_DIR}/svg`) ? readdirSync(`${ANIMATION_DIR}/svg`).sort() : []
    if (!stems.length) {
      console.warn("warn: no frames yet; run: bun run animation-svg <emoji>")
      return
    }
    writeFileSync(OUT, `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Animation frames</title>
<style>
:root{--ink:#1b1f24;--dim:#656b73;--line:#e2e5e9;--panel:#f5f6f8;--bg:#fff}
@media(prefers-color-scheme:dark){:root{--ink:#e8eaed;--dim:#9aa0a6;--line:#2c3036;--panel:#1c1f24;--bg:#121417}}
*{box-sizing:border-box}body{font:14px/1.4 system-ui,sans-serif;color:var(--ink);background:var(--bg);margin:0;padding:24px}
h2{margin:24px 0 12px}h2 small{color:var(--dim);font-weight:400;font-size:13px}
.row{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
.loop{position:sticky;top:12px;width:200px;text-align:center}.loop img{width:200px;height:200px;background:var(--panel);border:1px solid var(--line);border-radius:12px}
.loop label{color:var(--dim);font-size:12px}.fps{width:52px}
.grid{flex:1;min-width:300px;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}
figure img,.missing{width:100%;aspect-ratio:1;display:block}.missing{display:grid;place-items:center;color:var(--dim)}
figcaption{font-size:11px;color:var(--dim);margin-top:4px}figcaption b{color:var(--ink)}
</style></head><body>
${stems.map(section).join("\n")}
<script>
for (const s of document.querySelectorAll('section')) {
  const frames = [...s.querySelectorAll('figure img')].map(i => i.getAttribute('src')), live = s.querySelector('.live'), fps = s.querySelector('.fps')
  let k = 0, t
  const run = () => { clearInterval(t); t = setInterval(() => { k = (k + 1) % frames.length; live.src = frames[k] }, 1000 / Math.max(1, +fps.value || 12)) }
  fps.onchange = run; if (frames.length) run()
}
</script></body></html>
`)
    console.log(OUT)
    if (o.open) Bun.spawn(["xdg-open", OUT], { stdout: "ignore", stderr: "ignore" })
  })
cli.help()
cli.parse()
