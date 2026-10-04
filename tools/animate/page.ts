// Generates web/public/emoji-animation.html: static art next to the animation, for every animated
// emoji. Needs to be served over http (Lottie files are fetched).
import { existsSync, readFileSync, writeFileSync } from "node:fs"

import { EMOJI_ANIMATION_HTML, NOTO_LOTTIE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { readIndex } from "./index.ts"

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;")

export function writePage(): number {
  const byStem = new Map<string, string>() // stem -> emoji as labelled (shortest key)
  for (const [emoji, stem] of Object.entries(readIndex())) if (!byStem.has(stem) || emoji.length < byStem.get(stem)!.length) byStem.set(stem, emoji)
  const rows = [...byStem].filter(([stem]) => existsSync(`${NOTO_LOTTIE_DIR}/${stem}.json`)).map(([stem, emoji]) => ({ stem, emoji })).sort((a, b) => a.stem.localeCompare(b.stem))
  // Noto's own files are named "emoji_…"; the ones we generate are named after the emoji.
  const ours = (stem: string) => !/"nm":"emoji_/.test(readFileSync(`${NOTO_LOTTIE_DIR}/${stem}.json`, "utf8").slice(0, 500))
  const nOurs = rows.filter((r) => ours(r.stem)).length
  const cards = rows.map((r) => {
    const svg = resolveEmojiSvg(r.emoji)
    return `<div class="card${ours(r.stem) ? " ours" : ""}"><div class="pair">
<div class="cell"><svg viewBox="0 0 128 128">${svg?.body ?? ""}</svg><span>static</span></div>
<div class="cell"><div class="lot" data-src="noto/${r.stem}.json"></div><span>animated</span></div></div>
<div class="meta"><b>${esc(r.emoji)}</b> ${esc(r.stem)}</div></div>`
  })
  writeFileSync(EMOJI_ANIMATION_HTML, `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Animated emojis</title>
<style>
:root{--ink:#1b1f24;--dim:#656b73;--line:#e2e5e9;--panel:#f5f6f8}
@media(prefers-color-scheme:dark){:root{--ink:#e8eaed;--dim:#9aa0a6;--line:#2c3036;--panel:#1c1f24}body{background:#121417}}
*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;color:var(--ink);margin:0;background:#fff}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}h1{font-size:24px;margin:0 0 4px}.sub{color:var(--dim);margin:0 0 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.cell{display:flex;flex-direction:column;align-items:center;gap:4px;color:var(--dim);font-size:12px}
.cell svg,.lot{width:100%;aspect-ratio:1}.meta{margin-top:8px;font-size:12px;color:var(--dim)}.meta b{font-size:18px}
.sw{display:flex;gap:8px;align-items:center;margin:0 0 16px;color:var(--dim)}.only-ours .card:not(.ours){display:none}
</style></head><body><div class="wrap">
<h1>Animated emojis</h1>
<p class="sub">${rows.length} animated emoji. Left: static art. Right: the Lottie animation.</p>
<label class="sw"><input type="checkbox" id="ours"> Only our animations (${nOurs})</label>
<div class="grid" id="grid">${cards.join("\n")}</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
<script>document.querySelectorAll('.lot').forEach(el=>lottie.loadAnimation({container:el,renderer:'svg',loop:true,autoplay:true,path:el.dataset.src}))
const cb=document.getElementById('ours'),g=document.getElementById('grid');cb.onchange=()=>{g.classList.toggle('only-ours',cb.checked);try{localStorage.ours=cb.checked?1:''}catch{}};try{cb.checked=!!localStorage.ours}catch{};cb.onchange()</script>
</body></html>
`)
  return rows.length
}
