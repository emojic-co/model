import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"

import { EVAL_JSONL, STYLE_SAMPLES_JSON, TRAIN_JSONL } from "../../files.ts"
import { FEELINGS } from "../../web/src/feelings.js"

type Row = {
  text: string
  emojis: string
  styles: string[]
  colors: { bg: [string, string]; fg: string }[]
  extra?: { lang?: string }
}

type Sample = { text: string; emoji: string; colors: { bg1: string; bg2: string; text_color: string } }

function firstEmoji(field: string): string {
  return field.trim().split(/\s+/)[0] ?? ""
}

function toSample(row: Row): Sample {
  const [bg1, bg2] = row.colors[0].bg
  return { text: row.text, emoji: firstEmoji(row.emojis), colors: { bg1, bg2, text_color: row.colors[0].fg } }
}

async function loadRows(path: string): Promise<Row[]> {
  const text = await Bun.file(path).text()
  return text
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l))
}

function pick(rows: Row[], style: string, lang: "en" | "he"): Row | undefined {
  return rows.find((r) => r.styles[0] === style && (r.extra?.lang ?? "en") === lang)
}

const NEUTRAL_SAMPLES: Record<"en" | "he", Sample> = {
  en: { text: "Just got back from the store", emoji: "🙂", colors: { bg1: "#a8e2f4", bg2: "#78c9f4", text_color: "#282e36" } },
  he: { text: "עוד יום רגיל בעבודה", emoji: "🙂", colors: { bg1: "#a8e2f4", bg2: "#78c9f4", text_color: "#282e36" } },
}

async function main() {
  const evalRows = await loadRows(EVAL_JSONL)
  const trainRows = await loadRows(TRAIN_JSONL)
  const styles = Object.keys(FEELINGS)
  const out: Record<string, Record<"en" | "he", Sample>> = {}

  for (const style of styles) {
    if (style === "Neutral") {
      out[style] = NEUTRAL_SAMPLES
      continue
    }
    const en = pick(evalRows, style, "en") ?? pick(trainRows, style, "en")
    const he = pick(evalRows, style, "he") ?? pick(trainRows, style, "he")
    if (!en || !he) throw new Error(`no ${!en ? "en" : "he"} eval/train example found for style ${style}`)
    out[style] = { en: toSample(en), he: toSample(he) }
  }

  mkdirSync(dirname(STYLE_SAMPLES_JSON), { recursive: true })
  writeFileSync(STYLE_SAMPLES_JSON, JSON.stringify(out, null, 2) + "\n", "utf-8")
  console.log(`wrote ${STYLE_SAMPLES_JSON}`)
}

if (import.meta.main) main()
