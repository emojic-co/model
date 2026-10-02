// Shared bits of the SVG-frame animation tools: paths, emoji resolution, emoji metadata, one gateway call.
import { readFileSync } from "node:fs"

import { generateText } from "ai"

import { ANIMATION_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"

export const FRAMES = 20
export const jsonPath = (stem: string) => `${ANIMATION_DIR}/json/${stem}.json`
export const svgDir = (stem: string) => `${ANIMATION_DIR}/svg/${stem}`
export const framePath = (stem: string, n: number) => `${svgDir(stem)}/frame_${String(n).padStart(2, "0")}.svg`

/** Emoji code used in file names: hex code points joined by "_", VS16 dropped (e.g. 1f476, 2764_200d_1f525). */
export const stemOf = (emoji: string) =>
  [...emoji].map((c) => c.codePointAt(0)!).filter((cp) => cp !== 0xfe0f).map((cp) => cp.toString(16)).join("_")

/** CLI argument -> emoji: either the emoji itself or its code (`1f476`, `2764_200d_1f525`). */
export const parseEmoji = (arg: string) =>
  /^[0-9a-f]{4,6}([_-][0-9a-f]{4,6})*$/i.test(arg) ? arg.split(/[_-]/).map((h) => String.fromCodePoint(parseInt(h, 16))).join("") : arg

export function emojiSvg(emoji: string): string {
  const svg = resolveEmojiSvg(emoji)
  if (!svg) throw new Error(`no SVG for ${emoji}`)
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${svg.width} ${svg.height}">${svg.body}</svg>`
}

type Emojibase = { emoji: string; label: string; tags?: string[]; group?: number; subgroup?: number }
type Messages = { groups: { order: number; message: string }[]; subgroups: { order: number; message: string }[] }
const dir = "node_modules/emojibase-data/en"

/** Name, category and keywords of the emoji (emojibase). */
export function emojiMeta(emoji: string): string {
  const bare = (e: string) => e.replace(/️/g, "")
  const data: Emojibase[] = JSON.parse(readFileSync(`${dir}/data.json`, "utf8"))
  const msg: Messages = JSON.parse(readFileSync(`${dir}/messages.json`, "utf8"))
  const e = data.find((d) => bare(d.emoji) === bare(emoji))
  if (!e) return `Emoji: ${emoji}`
  const group = msg.groups.find((g) => g.order === e.group)?.message
  const sub = msg.subgroups.find((g) => g.order === e.subgroup)?.message
  return `Emoji: ${emoji}\nName: ${e.label}\nCategory: ${[group, sub].filter(Boolean).join(" / ")}\nKeywords: ${(e.tags ?? []).join(", ")}`
}

/** One call through the Vercel AI Gateway (key AI_GATEWAY_API_KEY, from .env). Never calls a provider directly. */
export async function ask(model: string, system: string, user: string, tag: string, extra: Record<string, unknown> = {}) {
  const t0 = Date.now()
  console.error(`-> ${tag} (${model}) ...`)
  const res = await generateText({ model, system, prompt: user, maxRetries: 1, abortSignal: AbortSignal.timeout(600_000), ...extra })
  console.error(`<- ${tag}: ${((Date.now() - t0) / 1000).toFixed(1)}s, ${res.usage.inputTokens ?? 0} in / ${res.usage.outputTokens ?? 0} out tokens`)
  return res
}
