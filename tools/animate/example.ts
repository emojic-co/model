// A worked example for the prompt: the nearest emoji (by EmojiHead embedding, see tools/animate_neighbors.py) that
// has an original Noto animation, as its full static SVG plus Noto's full Lottie JSON.
import { existsSync, readFileSync } from "node:fs"

import { ANIMATION_NEIGHBORS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"

export function exampleText(emoji: string): string | undefined {
  if (!existsSync(ANIMATION_NEIGHBORS_JSON)) return undefined
  const neighbors: Record<string, string[]> = JSON.parse(readFileSync(ANIMATION_NEIGHBORS_JSON, "utf8"))
  const index: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
  for (const n of neighbors[emoji] ?? []) {
    const svg = resolveEmojiSvg(n)
    const stem = index[n]
    if (!svg || !stem) continue
    const lottie = readFileSync(`${NOTO_LOTTIE_DIR}/${stem}.json`, "utf8")
    return `For reference, here is a related emoji, ${n}, with an original animation made by Google's Noto team.

${n} static SVG:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svg.width} ${svg.height}">${svg.body}</svg>

${n} animation (Lottie JSON):
${lottie}`
  }
  return undefined
}
