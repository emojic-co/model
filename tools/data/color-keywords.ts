import { readFile } from "node:fs/promises"

import { parse as parseYaml } from "yaml"

import { COLOR_KEYWORDS_JSONL, GOALS_YML } from "../../files.ts"
import { writeFileAtomic } from "./io.ts"

export type Palette = { bg: string[]; fg: string }
export type ColorRecord = { text: string; colors?: Palette[] }
export type ColorKeywordRow = { keyword: string; text: string; colors: Palette[] }

export async function loadColorEnergyKeywords(): Promise<string[]> {
  const doc = parseYaml(await readFile(GOALS_YML, "utf8")) as {
    goals?: Record<string, unknown>
  }
  const colorGen = (doc.goals?.["color generator"] ?? {}) as Record<string, unknown>
  const energy = (colorGen["energy distance"] ?? {}) as Record<string, number>
  return Object.keys(energy)
}

export function extractColorKeywords(
  records: ColorRecord[],
  keywords: string[],
): ColorKeywordRow[] {
  const out: ColorKeywordRow[] = []
  for (const r of records) {
    if (!r.colors || !r.colors.length) continue
    const padded = ` ${r.text.toLowerCase()} `
    for (const keyword of keywords) {
      if (padded.includes(keyword.toLowerCase())) {
        out.push({ keyword, text: r.text, colors: r.colors })
      }
    }
  }
  return out
}

function toLine(r: ColorKeywordRow): string {
  return JSON.stringify(r)
}

export async function writeColorKeywords(
  records: ColorRecord[],
): Promise<ColorKeywordRow[]> {
  const keywords = await loadColorEnergyKeywords()
  const rows = extractColorKeywords(records, keywords)
  await writeFileAtomic(COLOR_KEYWORDS_JSONL, rows.map(toLine).join("\n") + "\n")
  return rows
}
