// The animated-emoji collection: web/public/noto/index.json maps emoji -> Lottie file stem (Noto's own
// animations and the ones this tool generates). The Android copy of the collection is kept identical.
import { readFileSync, writeFileSync } from "node:fs"

import { ANDROID_ASSETS_DIR, LABELS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"

export const stemOf = (emoji: string) =>
  [...emoji].map((c) => c.codePointAt(0)!).filter((cp) => cp !== 0xfe0f).map((cp) => cp.toString(16)).join("_")

/** index.json keys for an emoji: as labelled, without VS16, and with VS16. */
export const indexKeys = (emoji: string): string[] => {
  const bare = emoji.replace(/️/g, "")
  return [...new Set([emoji, bare, `${bare}️`])]
}

export const readIndex = (): Record<string, string> => JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))

export const isAnimated = (emoji: string, idx = readIndex()) => indexKeys(emoji).some((k) => k in idx)

/** Every labelled emoji that has no animation yet. */
export function missing(): string[] {
  const labels: { emojis: string[] } = JSON.parse(readFileSync(LABELS_JSON, "utf8"))
  const idx = readIndex()
  return labels.emojis.filter((e) => !isAnimated(e, idx))
}

/** Adds the Lottie JSON for an emoji to the collection (web + Android). */
export function addAnimation(emoji: string, json: string): string {
  const stem = stemOf(emoji)
  const idx = readIndex()
  for (const k of indexKeys(emoji)) idx[k] = stem
  const sorted = JSON.stringify(Object.fromEntries(Object.entries(idx).sort(([a], [b]) => a.localeCompare(b))))
  for (const dir of [NOTO_LOTTIE_DIR, `${ANDROID_ASSETS_DIR}/noto`]) {
    writeFileSync(`${dir}/${stem}.json`, json)
    writeFileSync(`${dir}/index.json`, sorted)
  }
  return stem
}
