import { loadCldrAnnotations } from "../data/cldr.ts"

const MIN_LENGTHS = [5, 6, 7, 8]
const MAX_LENGTHS = [8, 9, 10, 11]

export function maxIdfKeywords(annotations: Map<string, string[]>): string[] {
  const index = new Map<string, Set<string>>()
  for (const [glyph, keywords] of annotations) {
    for (const raw of keywords) {
      const keyword = raw.trim()
      if (!keyword || /\s/.test(keyword)) continue
      let set = index.get(keyword)
      if (!set) index.set(keyword, (set = new Set()))
      set.add(glyph)
    }
  }
  return [...index.entries()].filter(([, set]) => set.size === 1).map(([kw]) => kw)
}

export function lengthMatrix(
  keywords: string[],
  minLengths: number[] = MIN_LENGTHS,
  maxLengths: number[] = MAX_LENGTHS,
): number[][] {
  const lens = keywords.map((k) => k.length)
  return minLengths.map((min) =>
    maxLengths.map((max) => lens.filter((n) => n >= min && n <= max).length),
  )
}

if (import.meta.main) {
  const annotations = await loadCldrAnnotations()
  const keywords = maxIdfKeywords(annotations)
  const matrix = lengthMatrix(keywords)

  console.log(
    `${annotations.size} CLDR emoji -> ${keywords.length} max-IDF keywords `
    + `(each appears in exactly one emoji)\n`,
  )
  const cell = (s: string) => s.padStart(8)
  console.log([cell("min\\max"), ...MAX_LENGTHS.map((m) => cell(`${m}`))].join(" "))
  MIN_LENGTHS.forEach((min, i) => {
    console.log([cell(`${min}`), ...matrix[i].map((n) => cell(`${n}`))].join(" "))
  })
}
