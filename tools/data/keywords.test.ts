import { expect, test } from "bun:test"

import { mergeKeywords, randomPalette, wordCount } from "./keywords.ts"

test("mergeKeywords unions emojis/styles and tags src by which source(s) had the keyword", () => {
  const cldr = [
    { text: "cake", emojis: "🍰", styles: ["Joyful"] },
    { text: "shared", emojis: "🍕 🧀", styles: ["Deadpan"] },
  ]
  const emojilib = [
    { text: "shared", emojis: "🧀 🍰", styles: ["Excited"] },
    { text: "rocket", emojis: "🚀", styles: ["Excited"] },
  ]
  expect(mergeKeywords(cldr, emojilib)).toEqual([
    { text: "cake", emojis: ["🍰"], styles: ["Joyful"], src: "cldr" },
    { text: "rocket", emojis: ["🚀"], styles: ["Excited"], src: "emojilib" },
    {
      text: "shared",
      emojis: ["🍕", "🍰", "🧀"],
      styles: ["Deadpan", "Excited"],
      src: "both",
    },
  ])
})

test("randomPalette draws bg/fg from the fallback palette pool", () => {
  const p = randomPalette(() => 0)
  expect(p.bg).toHaveLength(2)
  expect(typeof p.fg).toBe("string")
})

test("wordCount counts whitespace-separated words", () => {
  expect(wordCount("cake")).toBe(1)
  expect(wordCount("thumbs up")).toBe(2)
  expect(wordCount("  face palm emoji  ")).toBe(3)
  expect(wordCount("!!")).toBe(1)
})
