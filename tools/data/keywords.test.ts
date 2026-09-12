import { expect, test } from "bun:test"

import { mergeKeywords, splitKeywordsAndTerms, wordCount } from "./keywords.ts"

const P = (a: string, b: string, f: string): { bg: [string, string]; fg: string } => ({
  bg: [a, b],
  fg: f,
})

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

test("mergeKeywords picks a random palette from among the source(s) that had one", () => {
  const cldr = [{ text: "sun", emojis: "☀️", styles: [], ...P("#111111", "#222222", "#eeeeee") }]
  const emojilib = [{ text: "sun", emojis: "☀️", styles: [], ...P("#333333", "#444444", "#dddddd") }]
  const first = mergeKeywords(cldr, emojilib, () => 0)[0]
  expect(first.bg).toEqual(["#111111", "#222222"])
  expect(first.fg).toBe("#eeeeee")
  const second = mergeKeywords(cldr, emojilib, () => 0.999)[0]
  expect(second.bg).toEqual(["#333333", "#444444"])
  expect(second.fg).toBe("#dddddd")
})

test("mergeKeywords omits bg/fg when neither source had a palette", () => {
  const out = mergeKeywords([{ text: "cake", emojis: "🍰", styles: [] }], [])
  expect("bg" in out[0]).toBe(false)
  expect("fg" in out[0]).toBe(false)
})

test("wordCount counts whitespace-separated words", () => {
  expect(wordCount("cake")).toBe(1)
  expect(wordCount("thumbs up")).toBe(2)
  expect(wordCount("  face palm emoji  ")).toBe(3)
  expect(wordCount("!!")).toBe(1)
})

test("splitKeywordsAndTerms routes single-word rows to keywords, everything else to terms", () => {
  const merged = mergeKeywords(
    [
      { text: "cake", emojis: "🍰 🎂", styles: [] },
      { text: "jazz hands", emojis: "🙌", styles: [] },
    ],
    [],
  )
  const { keywords, terms } = splitKeywordsAndTerms(merged, new Set(["🍰", "🙌"]))
  expect(keywords.map((r) => r.text)).toEqual(["cake"])
  expect(keywords[0].emojis).toEqual(["🍰"])
  expect(terms.map((r) => r.text)).toEqual(["jazz hands"])
})

test("splitKeywordsAndTerms drops rows with no in-vocab emoji", () => {
  const merged = mergeKeywords([{ text: "cake", emojis: "🍰", styles: [] }], [])
  const { keywords, terms } = splitKeywordsAndTerms(merged, new Set(["🚀"]))
  expect(keywords).toEqual([])
  expect(terms).toEqual([])
})
