import { readFile } from "node:fs/promises"

import { Document } from "flexsearch"

const CLDR_ANNOTATIONS = "node_modules/cldr-annotations-full/annotations/en/annotations.json"
const CLDR_ANNOTATIONS_DERIVED =
  "node_modules/cldr-annotations-derived-full/annotationsDerived/en/annotations.json"
const EMOJIBASE_DATA = "node_modules/emojibase-data/en/data.json"

type Annotation = { default?: string[]; tts?: string[] }
type CldrAnnotations = { annotations: { annotations: Record<string, Annotation> } }
type CldrAnnotationsDerived = { annotationsDerived: { annotations: Record<string, Annotation> } }
type EmojibaseEntry = { emoji: string }

export interface CldrEmoji {
  id: string
  name: string
  keywords: string
  [key: string]: string
}

const VARIATION_SELECTOR = /️/g
const stripVariationSelector = (emoji: string) => emoji.replace(VARIATION_SELECTOR, "")

async function loadCldrEmoji(): Promise<CldrEmoji[]> {
  const [cldr, cldrDerived, emojibase] = await Promise.all([
    readFile(CLDR_ANNOTATIONS, "utf8").then((s) => JSON.parse(s) as CldrAnnotations),
    readFile(CLDR_ANNOTATIONS_DERIVED, "utf8").then(
      (s) => JSON.parse(s) as CldrAnnotationsDerived,
    ),
    readFile(EMOJIBASE_DATA, "utf8").then((s) => JSON.parse(s) as EmojibaseEntry[]),
  ])
  // CLDR's own annotation keys often omit the U+FE0F variation selector
  // (e.g. "🌧" not "🌧️") while emojibase's canonical form includes it, so
  // match with the selector stripped but keep emojibase's canonical glyph.
  const canonicalById = new Map(emojibase.map((e) => [stripVariationSelector(e.emoji), e.emoji]))
  const annotations = {
    ...cldr.annotations.annotations,
    ...cldrDerived.annotationsDerived.annotations,
  }
  const entries: CldrEmoji[] = []
  for (const [key, { default: keywords = [], tts = [] }] of Object.entries(annotations)) {
    const id = canonicalById.get(stripVariationSelector(key))
    if (!id) continue
    entries.push({ id, name: tts[0] ?? "", keywords: keywords.join(" ") })
  }
  return entries.sort((a, b) => a.id.codePointAt(0)! - b.id.codePointAt(0)!)
}

const cldrData = await loadCldrEmoji()

const emojiIndex = new Document<CldrEmoji, true>({
  document: {
    id: "id",
    index: ["name", "keywords"],
  },
  tokenize: "strict",
})
cldrData.forEach((emoji) => emojiIndex.add(emoji))

export async function cldrEmojis(text: string, limit = 10): Promise<string[]> {
  if (!text.trim()) return []

  const rawResults = await emojiIndex.search(text, { limit, enrich: false, suggest: true })

  const matchedEmojis = new Set<string>()
  for (const fieldResult of rawResults) {
    for (const emojiId of fieldResult.result) {
      matchedEmojis.add(emojiId as string)
    }
  }
  return Array.from(matchedEmojis).slice(0, limit)
}

if (import.meta.main) {
  console.log(`loaded ${cldrData.length} CLDR emoji`)
  console.log("spicy ->", cldrEmojis("spicy"))
  console.log("tired ->", cldrEmojis("tired"))
  console.log("happy birthday ->", cldrEmojis("happy birthday"))
}
