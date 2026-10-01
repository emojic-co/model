// Headless-Chrome contact sheets: the original SVG next to evenly spaced Lottie frames, so a
// vision model (or a human) can judge the animation without a player.
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"

const LOTTIE_JS = resolve("web/node_modules/lottie-web/build/player/lottie.min.js")
const CHROME = process.env.CHROME ?? ["google-chrome", "chromium", "chromium-browser"].find((b) => Bun.which(b))

const CELL = 220

async function shoot(html: string, w: number, h: number): Promise<Buffer> {
  if (!CHROME) throw new Error("no Chrome/Chromium found (set CHROME=/path/to/chrome)")
  const dir = mkdtempSync(join(tmpdir(), "emoji-anim-"))
  const page = join(dir, "sheet.html")
  const png = join(dir, "sheet.png")
  writeFileSync(page, html)
  const p = Bun.spawn(
    [CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--window-size=${w},${h}`,
      "--allow-file-access-from-files", "--virtual-time-budget=4000", `--screenshot=${png}`, `file://${page}`],
    { stdout: "ignore", stderr: "ignore" },
  )
  await p.exited
  return readFileSync(png)
}

const page = (cells: string, script: string, cols: number) => `<!doctype html><body style="margin:0;background:#2a2a40;display:grid;
grid-template-columns:repeat(${cols},${CELL}px);gap:4px;font:12px sans-serif;color:#fff">${cells}
<script>${readFileSync(LOTTIE_JS, "utf8")}</script><script>${script}</script>`

const svgCell = (body: string, label: string) =>
  `<div><svg viewBox="0 0 128 128" width="${CELL}" height="${CELL}">${body}</svg>${label}</div>`

export const staticPng = (body: string) => shoot(page(svgCell(body, "static"), "", 1), CELL + 4, CELL + 24)

export function sheetPng(body: string, lottie: object, frames: number, n = 11): Promise<Buffer> {
  const at = Array.from({ length: n }, (_, k) => Math.round((k * frames) / n))
  const cells = at.map((f) => `<div><div id="f${f}" style="width:${CELL}px;height:${CELL}px"></div>frame ${f}/${frames}</div>`).join("")
  const script = `const d=${JSON.stringify(lottie)};${JSON.stringify(at)}.forEach(f=>{
    const a=lottie.loadAnimation({container:document.getElementById('f'+f),renderer:'svg',loop:false,autoplay:false,animationData:d});
    a.addEventListener('DOMLoaded',()=>a.goToAndStop(f,true));});`
  const cols = 6
  const rows = Math.ceil((n + 1) / cols)
  return shoot(page(svgCell(body, "original") + cells, script, cols), cols * (CELL + 4), rows * (CELL + 24))
}
