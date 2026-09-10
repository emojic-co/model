import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"

import { II_JSON, REPORT_DIR } from "../../files.ts"
import { queryTokens } from "../analysis/cldr-baseline.ts"

const CAP = 1000

function isCandidate(kw: string): boolean {
  if (kw.length < 3 || kw.length > 6) return false
  const qt = queryTokens(kw)
  return qt.length === 1 && qt[0] === kw
}

export function keywordVocabFromIiJson(emojiVocab: string[]): string[] {
  const emojis = new Set(emojiVocab)
  const ii = JSON.parse(readFileSync(II_JSON, "utf8")) as Record<string, string[]>
  const out: string[] = []
  for (const [kw, es] of Object.entries(ii)) {
    if (!isCandidate(kw)) continue
    if (!es.some((e) => emojis.has(e))) continue
    out.push(kw)
  }
  return [...new Set(out)].sort().slice(0, CAP)
}

export function keywordVocabFromReport(dir: string = REPORT_DIR): string[] | null {
  if (!existsSync(dir)) return null
  const folders = readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .sort()
  for (let i = folders.length - 1; i >= 0; i--) {
    const p = join(dir, folders[i], "report.json")
    if (!existsSync(p)) continue
    let ranked: { kw: string; rank: number }[] | undefined
    try {
      const parsed = JSON.parse(readFileSync(p, "utf8")) as {
        keywords_flex?: { ranked?: { kw: string; rank: number }[] }
      }
      ranked = parsed.keywords_flex?.ranked
    } catch {
      return null
    }
    if (!ranked || ranked.length === 0) return null
    return ranked
      .slice(0, CAP)
      .map((r) => r.kw)
      .sort()
  }
  return null
}
