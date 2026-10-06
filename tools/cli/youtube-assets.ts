// Renders YouTube channel art (banner 2560x1440, profile 800x800, watermark 150x150) into pr/common/.
// Usage: bun tools/cli/youtube-assets.ts
import { ensureFonts, FONTS_XDG_DATA_HOME } from "./fonts.ts"
import { resolveEmojiSvg } from "./emoji-svg.ts"

const OUT = "pr/common"
const FACE = "🙂"

function emoji(x: number, y: number, size: number): string {
  const e = resolveEmojiSvg(FACE)
  if (!e) throw new Error("emoji not found")
  const s = size / Math.max(e.width, e.height)
  return `<g filter="url(#shadow)"><g transform="translate(${x} ${y}) scale(${s})">${e.body}</g></g>`
}

function defs(w: number, h: number): string {
  return `<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffd36b"/><stop offset="1" stop-color="#ffab26"/></linearGradient>
  <pattern id="dots" width="${w / 50}" height="${w / 50}" patternUnits="userSpaceOnUse"><circle cx="${w / 100}" cy="${w / 100}" r="${w / 700}" fill="#fff" fill-opacity="0.35"/></pattern>
  <filter id="shadow" filterUnits="userSpaceOnUse" x="0" y="0" width="${w}" height="${h}"><feDropShadow dx="0" dy="${w / 90}" stdDeviation="${w / 70}" flood-color="#8a4b00" flood-opacity="0.22"/></filter>
</defs>
<rect width="${w}" height="${h}" fill="url(#bg)"/><rect width="${w}" height="${h}" fill="url(#dots)"/>`
}

const font = (weight: number) => `font-family="Montserrat, Noto Sans, sans-serif" font-weight="${weight}"`

// Safe area for all devices is the central 1546x423 of the 2560x1440 canvas.
function banner(): string {
  const w = 2560, h = 1440, size = 300
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
${defs(w, h)}
${emoji(w / 2 - 640, h / 2 - size / 2, size)}
<text x="${w / 2 - 290}" y="${h / 2 + 30}" ${font(800)} font-size="120" letter-spacing="11" fill="#3a2406">EMOJIFY<tspan fill="#3a2406" fill-opacity="0.55">.ING</tspan></text>
<text x="${w / 2 - 287}" y="${h / 2 + 100}" ${font(600)} font-size="42" letter-spacing="5" fill="#3a2406" fill-opacity="0.8">TEXT → EMOJI, STYLE &amp; COLORS</text>
</svg>`
}

// Shown as a circle, so keep the face well inside.
function profile(): string {
  const w = 800, size = 520
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${w}" viewBox="0 0 ${w} ${w}">
${defs(w, w)}
${emoji((w - size) / 2, (w - size) / 2 - 10, size)}
</svg>`
}

// Transparent, so it blends over video; no shadow (it is tiny).
function watermark(): string {
  const e = resolveEmojiSvg(FACE)
  if (!e) throw new Error("emoji not found")
  const s = 150 / Math.max(e.width, e.height)
  return `<svg xmlns="http://www.w3.org/2000/svg" width="150" height="150" viewBox="0 0 150 150"><g transform="scale(${s})">${e.body}</g></svg>`
}

async function render(svg: string, dest: string): Promise<void> {
  const proc = Bun.spawn(["rsvg-convert", "-f", "png", "-"], {
    stdin: "pipe", stdout: "pipe", stderr: "pipe",
    env: { ...process.env, XDG_DATA_HOME: FONTS_XDG_DATA_HOME },
  })
  proc.stdin.write(svg)
  proc.stdin.end()
  const [png, code] = await Promise.all([new Response(proc.stdout).arrayBuffer(), proc.exited])
  if (code !== 0) throw new Error(await new Response(proc.stderr).text())
  await Bun.write(dest, png)
}

await ensureFonts(["family=Montserrat:wght@600;800"])
await render(banner(), `${OUT}/youtube-banner.png`)
await render(profile(), `${OUT}/youtube-profile.png`)
await render(watermark(), `${OUT}/youtube-watermark.png`)
