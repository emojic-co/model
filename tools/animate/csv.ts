// animation.csv: one row per labelled emoji. `noto` = Google ships an animation for it; for the rest,
// `complexity`/`unsupported` say how animatable its SVG is and `animation_text` (tools/animate/describe.ts)
// is the brief for the LLM. Whether we generated a Lottie is not stored: see hasLottie.
import { existsSync, readFileSync, writeFileSync } from "node:fs"

import { ANIMATION_CSV, LABELS_JSON, NOTO_LOTTIE_DIR } from "../../files.ts"
import { resolveEmojiSvg } from "../cli/emoji-svg.ts"
import { complexity, parseScene } from "./svg.ts"

export const COLUMNS = ["emoji", "name", "noto", "complexity", "unsupported", "animation_text"] as const
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
  if (!existsSync(ANIMATION_CSV)) return []
  const text = readFileSync(ANIMATION_CSV, "utf8")
  const [head, ...lines] = splitRecords(text)
  const cols = splitLine(head)
  return lines.map((l) => {
    const v = splitLine(l)
    return Object.fromEntries(COLUMNS.map((c) => [c, v[cols.indexOf(c)] ?? ""])) as Row
  })
}

/** Records are newline-separated, except inside quotes (animation_text may hold line breaks). */
function splitRecords(text: string): string[] {
  const out: string[] = []
  let cur = "", quoted = false
  for (const c of text) {
    if (c === '"') quoted = !quoted
    if (c === "\n" && !quoted) { out.push(cur); cur = "" } else cur += c
  }
  if (cur) out.push(cur)
  return out.filter(Boolean)
}

export function writeRows(rows: Row[]): void {
  writeFileSync(ANIMATION_CSV, [COLUMNS.join(","), ...rows.map((r) => COLUMNS.map((c) => q(r[c])).join(","))].join("\n") + "\n")
}

export const stemOf = (emoji: string) =>
  [...emoji].map((c) => c.codePointAt(0)!).filter((cp) => cp !== 0xfe0f).map((cp) => cp.toString(16)).join("_")

/** Did we generate a Lottie for this emoji? (the file's existence is the only record) */
export const hasLottie = (r: Row) => !r.noto && existsSync(`${NOTO_LOTTIE_DIR}/${stemOf(r.emoji)}.json`)

/** index.json keys for an emoji: as labelled, without VS16, and with VS16. */
export const indexKeys = (emoji: string): string[] => {
  const bare = emoji.replace(/\uFE0F/g, "")
  return [...new Set([emoji, bare, `${bare}\uFE0F`])]
}

export function measure(emoji: string): Pick<Row, "complexity" | "unsupported"> {
  const svg = resolveEmojiSvg(emoji)
  if (!svg) return { complexity: "", unsupported: "no-svg" }
  const sc = parseScene(svg.body, svg.width)
  return { complexity: String(complexity(sc)), unsupported: sc.unsupported.join(";") }
}

const bare = (e: string) => e.replace(/\uFE0F/g, "")

/** Rebuilds the CSV from the label vocab. Keeps `noto` and `animation_text` of existing rows; a new emoji is Noto's if Noto's index has it. */
export function refreshRows(): Row[] {
  const labels: { emojis: string[] } = JSON.parse(readFileSync(LABELS_JSON, "utf8"))
  const index: Record<string, string> = JSON.parse(readFileSync(`${NOTO_LOTTIE_DIR}/index.json`, "utf8"))
  const names: { emoji: string; label: string }[] = JSON.parse(readFileSync("node_modules/emojibase-data/en/data.json", "utf8"))
  const nameOf = new Map(names.map((n) => [bare(n.emoji), n.label]))
  const old = new Map(readRows().map((r) => [r.emoji, r]))
  return labels.emojis.map((emoji) => {
    const prev = old.get(emoji)
    const noto = prev ? prev.noto : indexKeys(emoji).some((k) => k in index) ? "1" : ""
    const meta = noto ? { complexity: "", unsupported: "" } : measure(emoji)
    return { emoji, name: nameOf.get(bare(emoji)) ?? "", noto, ...meta, animation_text: prev?.animation_text ?? "" }
  })
}
