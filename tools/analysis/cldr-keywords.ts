import { loadCldrAnnotations } from "../data/cldr.ts"
import { queryTokens } from "./cldr-baseline.ts"

const MIN_ROWS = [4, 5, 6, 7, 8]
const MAX_COLS = [6, 7, 8, 9, 10, 11]

function dfByKeyword(ann: Map<string, string[]>): Map<string, number> {
  const df = new Map<string, number>()
  for (const [, kws] of ann) {
    const uniq = new Set(kws.map((k) => k.trim().toLowerCase()).filter(Boolean))
    for (const k of uniq) df.set(k, (df.get(k) ?? 0) + 1)
  }
  return df
}

function matrix(words: string[]): string {
  const rng = (mn: number, mx: number) =>
    words.filter((w) => w.length >= mn && w.length <= mx).length
  const head = "        " + MAX_COLS.map((c) => `max=${c}`.padStart(8)).join("")
  const rows = MIN_ROWS.map(
    (mn) =>
      `min=${mn}  ` + MAX_COLS.map((mx) => String(rng(mn, mx)).padStart(8)).join(""),
  )
  return [head, ...rows].join("\n")
}

const ann = await loadCldrAnnotations()
const df = dfByKeyword(ann)
const maxIdf = [...df.values()].filter((n) => n === 1)
const all = [...df.entries()].filter(([, n]) => n === 1).map(([k]) => k)
const singleTok = all.filter((k) => {
  const qt = queryTokens(k)
  return qt.length === 1 && qt[0] === k
})

console.log(`CLDR keywords (default + derived + tts, trimmed + lowercased)`)
console.log(`  distinct keywords : ${df.size}`)
console.log(`  max-IDF (df == 1) : ${maxIdf.length}`)
console.log(``)
console.log(`max-IDF keyword count by [min length, max length] (chars):`)
console.log(``)
console.log(`all max-IDF keywords:`)
console.log(matrix(all))
console.log(``)
console.log(`single-token [a-z0-9]+ max-IDF keywords (fusion-vocab candidates):`)
console.log(matrix(singleTok))
