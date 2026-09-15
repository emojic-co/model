import { execFile } from "node:child_process"
import { existsSync } from "node:fs"
import { mkdir, readFile, writeFile } from "node:fs/promises"
import { resolve } from "node:path"
import { promisify } from "node:util"

const execFileAsync = promisify(execFile)

const CACHE_DIR = resolve("tools/cli/.cache")
export const FONTS_XDG_DATA_HOME = `${CACHE_DIR}/xdg`
const FONTS_DIR = `${FONTS_XDG_DATA_HOME}/fonts`

type Style = "normal" | "italic"
type FontFace = { family: string; style: Style; weight: number; url: string }

function parseFontFaces(css: string): FontFace[] {
  const faces: FontFace[] = []
  for (const block of css.matchAll(/@font-face\s*\{([^}]*)\}/g)) {
    const body = block[1]
    const family = body.match(/font-family:\s*'([^']+)'/)?.[1]
    const style = body.match(/font-style:\s*(\w+)/)?.[1] === "italic" ? "italic" : "normal"
    const weight = Number(body.match(/font-weight:\s*(\d+)/)?.[1] ?? "400")
    const url = body.match(/src:\s*url\(([^)]+)\)/)?.[1]
    if (family && url) faces.push({ family, style, weight, url })
  }
  return faces
}

function fileFor(face: Pick<FontFace, "family" | "style" | "weight">): string {
  const slug = `${face.family.replace(/\s+/g, "-")}-${face.weight}-${face.style}.ttf`
  return `${FONTS_DIR}/${slug}`
}

async function googleFontsUrl(): Promise<string> {
  const indexHtml = await readFile("web/index.html", "utf8")
  const m = indexHtml.match(/href="(https:\/\/fonts\.googleapis\.com\/[^"]+)"/)
  if (!m) throw new Error("fonts: no Google Fonts link found in web/index.html")
  return m[1]
}

export class FontCache {
  private byKey = new Map<string, string>()
  private byFamily = new Map<string, string>()

  add(face: FontFace, path: string): void {
    this.byKey.set(`${face.family}|${face.weight}|${face.style}`, path)
    if (!this.byFamily.has(face.family)) this.byFamily.set(face.family, path)
  }

  fileFor(family: string, weight: number, style: Style): string | undefined {
    return (
      this.byKey.get(`${family}|${weight}|${style}`)
      ?? this.byKey.get(`${family}|${weight}|normal`)
      ?? this.byFamily.get(family)
    )
  }
}

export async function ensureFonts(): Promise<FontCache> {
  await mkdir(FONTS_DIR, { recursive: true })
  const url = await googleFontsUrl()
  const css = await (await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } })).text()
  const faces = parseFontFaces(css)

  const cache = new FontCache()
  let downloaded = 0
  for (const face of faces) {
    const path = fileFor(face)
    cache.add(face, path)
    if (existsSync(path)) continue
    const res = await fetch(face.url)
    await writeFile(path, new Uint8Array(await res.arrayBuffer()))
    downloaded++
  }
  if (downloaded) await execFileAsync("fc-cache", ["-f", FONTS_DIR])

  return cache
}
