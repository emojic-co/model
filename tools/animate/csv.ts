// emoji_animation.csv: every labelled emoji without a Noto animation, its animation complexity, and
// (once generated) the Lottie file stem we made for it.
import { existsSync, readFileSync, writeFileSync } from "node:fs"

import { EMOJI_ANIMATION_CSV, LABELS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { complexity, parseScene } from "./svg.ts"

export const COLUMNS = ["emoji", "codepoints", "complexity", "parts", "nodes", "unsupported", "lottie", "model", "animated_at", "note"] as const
export type Row = Record<(typeof COLUMNS)[number], string>

const q = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)

function splitLine(line: string): string[] {
  const out: string[] = []
  let cur = "", quoted = false
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++ } else if (c === '"') quoted = false
      else cur += c
    } else if (c === '"') quoted = true
    else if (c === ",") { out.push(cur); cur = "" } else cur += c
  }
  return [...out, cur]
}

export function readRows(): Row[] {
  if (!existsSync(EMOJI_ANIMATION_CSV)) return []
  const [head, ...lines] = readFileSync(EMOJI_ANIMATION_CSV, "utf8").split("\n").filter(Boolean)
  const cols = splitLine(head)
  return lines.map((l) => {
    const v = splitLine(l)
    return Object.fromEntries(COLUMNS.map((c) => [c, v[cols.indexOf(c)] ?? ""])) as Row
  })
}

export function writeRows(rows: Row[]): void {
  writeFileSync(EMOJI_ANIMATION_CSV, [COLUMNS.join(","), ...rows.map((r) => COLUMNS.map((c) => q(r[c])).join(","))].join("\n") + "\n")
}

export const stemOf = (emoji: string) =>
  [...emoji].map((c) => c.codePointAt(0)!).filter((cp) => cp !== 0xfe0f).map((cp) => cp.toString(16)).join("_")

/** index.json keys for an emoji: as labelled, without VS16, and with VS16. */
export const indexKeys = (emoji: string): string[] => {
  const bare = emoji.replace(/\uFE0F/g, "")
  return [...new Set([emoji, bare, `${bare}\uFE0F`])]
}

export function measure(emoji: string): Pick<Row, "complexity" | "parts" | "nodes" | "unsupported"> {
  const svg = resolveEmojiSvg(emoji)
  if (!svg) return { complexity: "", parts: "", nodes: "", unsupported: "no-svg" }
  const sc = parseScene(svg.body, svg.width)
  return {
    complexity: String(complexity(sc)),
    parts: String(sc.parts.length),
    nodes: String(sc.parts.reduce((n, p) => n + p.nodes, 0)),
    unsupported: sc.unsupported.join(";"),
  }
}

/** Rebuilds the CSV: existing rows (keeps `lottie` etc.) + labelled emojis that still have no animation. */
export function refreshRows(): Row[] {
  const labels: { emojis: string[] } = JSON.parse(readFileSync(LABELS_JSON, "utf8"))
  const noto: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
  const old = new Map(readRows().map((r) => [r.emoji, r]))
  const rows: Row[] = []
  for (const emoji of labels.emojis) {
    const prev = old.get(emoji)
    if (!prev && emoji in noto) continue
    const base = prev ?? (Object.fromEntries(COLUMNS.map((c) => [c, ""])) as Row)
    rows.push({ ...base, emoji, codepoints: [...emoji].map((c) => c.codePointAt(0)!.toString(16)).join(" "), ...measure(emoji) })
  }
  rows.sort((a, b) => Number(a.complexity || 1e9) - Number(b.complexity || 1e9) || a.emoji.localeCompare(b.emoji))
  return rows
}
