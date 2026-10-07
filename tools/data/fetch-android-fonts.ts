// Bundles every card font into android/app/src/main/assets/fonts/<Family_Name>[_Italic].ttf, so the Android card never
// depends on the Google font provider (which can silently fall back to a system font). The family/weight/italic set is
// whatever the web loads: web/index.html (latin), fonts.yml (scripts + fallback). Existing files are kept; delete to refetch.
// Usage: bun run tools/data/fetch-android-fonts.ts
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { SCRIPT_FONT_QUERY, FONTS } from "../../web/src/scriptFonts.js"

const OUT = "android/app/src/main/assets/fonts"
const UA = "Mozilla/5.0 (Linux; U; Android 4.4; en-us) AppleWebKit/534.30" // old UA => static .ttf, no unicode-range subsets

const html = readFileSync("web/index.html", "utf-8")
const latinHref = html.match(/href="(https:\/\/fonts\.googleapis\.com\/css2\?[^"]+)"/g)!
  .map((h) => h.slice(6, -1).replaceAll("&amp;", "&")).find((h) => h.includes("Anton"))!
const queries = [
  latinHref,
  ...Object.values(SCRIPT_FONT_QUERY).map((q) => `https://fonts.googleapis.com/css2?${q}&display=swap`),
  `https://fonts.googleapis.com/css2?family=${FONTS.fallback.family.replaceAll(" ", "+")}&display=swap`,
]

mkdirSync(OUT, { recursive: true })
for (const url of queries) {
  const css = await (await fetch(url, { headers: { "User-Agent": UA } })).text()
  for (const [, body] of css.matchAll(/@font-face\s*{([^}]*)}/g)) {
    const family = body.match(/font-family:\s*'([^']+)'/)![1]
    if (family === "Noto Color Emoji") continue
    const italic = /font-style:\s*italic/.test(body)
    const src = body.match(/url\(([^)]+\.ttf)\)/)?.[1]
    if (!src) { console.warn("no ttf for", family); continue }
    const file = `${OUT}/${family.replaceAll(" ", "_")}${italic ? "_Italic" : ""}.ttf`
    if (existsSync(file)) continue
    writeFileSync(file, new Uint8Array(await (await fetch(src)).arrayBuffer()))
    console.log("wrote", file)
  }
}
