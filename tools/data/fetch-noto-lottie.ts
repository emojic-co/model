// Downloads Google's animated Noto emoji (Lottie, CC BY 4.0) for every emoji in the label vocab
// into web/public/noto/, plus index.json mapping emoji -> file stem. Emojis without an
// animated clone are simply absent; the web preview and Android app fall back to the static glyph.
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"

import { ANDROID_ASSETS_DIR, LABELS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"

const BASE = "https://fonts.gstatic.com/s/e/notoemoji/latest"
const VS16 = 0xfe0f

const stem = (emoji: string, keepVs: boolean) =>
  [...emoji]
    .map((c) => c.codePointAt(0)!)
    .filter((cp) => keepVs || cp !== VS16)
    .map((cp) => cp.toString(16))
    .join("_")

async function fetchLottie(emoji: string): Promise<{ stem: string; json: unknown } | null> {
  for (const keepVs of [true, false]) {
    const s = stem(emoji, keepVs)
    const res = await fetch(`${BASE}/${s}/lottie.json`)
    if (res.ok) return { stem: s, json: await res.json() }
  }
  return null
}

const labels = await Bun.file(LABELS_JSON).json()
const emojis: string[] = labels.emojis
mkdirSync(NOTO_LOTTIE_DIR, { recursive: true })

const index: Record<string, string> = {}
const queue = [...emojis]
await Promise.all(
  Array.from({ length: 16 }, async () => {
    for (let e = queue.pop(); e !== undefined; e = queue.pop()) {
      const hit = await fetchLottie(e)
      if (!hit) continue
      writeFileSync(`${NOTO_LOTTIE_DIR}/${hit.stem}.json`, JSON.stringify(hit.json))
      index[e] = hit.stem
    }
  }),
)

// Keep index entries for animations we added ourselves (their file is in the dir, Google's set doesn't have them).
const INDEX = `${NOTO_LOTTIE_DIR}/index.json`
const kept: Record<string, string> = existsSync(INDEX) ? JSON.parse(readFileSync(INDEX, "utf8")) : {}
for (const [k, v] of Object.entries(kept)) if (existsSync(`${NOTO_LOTTIE_DIR}/${v}.json`)) index[k] ??= v

const sorted = Object.fromEntries(Object.entries(index).sort(([a], [b]) => a.localeCompare(b)))
writeFileSync(`${NOTO_LOTTIE_DIR}/index.json`, JSON.stringify(sorted))
writeFileSync(
  `${NOTO_LOTTIE_DIR}/LICENSE.txt`,
  "Animated emoji by Google (Noto Emoji Animation), licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\nSource: https://googlefonts.github.io/noto-emoji-animation/\n",
)
cpSync(NOTO_LOTTIE_DIR, `${ANDROID_ASSETS_DIR}/noto`, { recursive: true })
console.log(`noto lottie: ${Object.keys(sorted).length}/${emojis.length} emojis`)
