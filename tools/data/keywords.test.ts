import { expect, test } from "bun:test"

import { buildFlags, mergeKeywords, splitKeywordsAndTerms, wordCount } from "./keywords.ts"

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

test("mergeKeywords merges rows that only differ pre-normalization", () => {
  const cldr = [{ text: "Christmas", emojis: "🎄 🤶", styles: ["Joyful"] }]
  const emojilib = [{ text: "christmas", emojis: "❄️ 🌲", styles: ["Excited"] }]
  expect(mergeKeywords(cldr, emojilib)).toEqual([
    {
      text: "christmas",
      emojis: ["❄️", "🌲", "🎄", "🤶"],
      styles: ["Joyful", "Excited"],
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

test("splitKeywordsAndTerms normalizes text and drops rows shorter than 3 chars once normalized", () => {
  const merged = mergeKeywords(
    [
      { text: "BACK Arrow", emojis: "🔙", styles: [] },
      { text: "24", emojis: "🏪", styles: [] },
      { text: "- -", emojis: "😑", styles: [] },
    ],
    [],
  )
  const { keywords, terms } = splitKeywordsAndTerms(merged, new Set(["🔙", "🏪", "😑"]))
  expect(terms.map((r) => r.text)).toEqual(["back arrow"])
  expect(keywords).toEqual([])
})

test("splitKeywordsAndTerms drops rows that are entirely flag emoji, single- or multi-word", () => {
  const merged = mergeKeywords(
    [
      { text: "Japan", emojis: "🇯🇵", styles: [] },
      { text: "United States", emojis: "🇺🇸", styles: [] },
      { text: "cake", emojis: "🍰", styles: [] },
    ],
    [],
  )
  const { keywords, terms } = splitKeywordsAndTerms(
    merged,
    new Set(["🇯🇵", "🇺🇸", "🍰"]),
  )
  expect(keywords.map((r) => r.text)).toEqual(["cake"])
  expect(terms).toEqual([])
})

test("buildFlags keeps cldr rows whose sole in-vocab emoji is a flag, keyed by normalized text", () => {
  const cldr = [
    { text: "Japan", emojis: "🇯🇵", styles: ["Deadpan"], ...P("#fffafa", "#d71920", "#17202a") },
    { text: "United States", emojis: "🇺🇸", styles: ["Deadpan"], ...P("#173f73", "#b22234", "#ffffff") },
    { text: "cake", emojis: "🍰", styles: [] },
    { text: "flag", emojis: "🇯🇵 🇺🇸 🏁", styles: ["Deadpan"], ...P("#111111", "#222222", "#eeeeee") },
  ]
  const flags = buildFlags(cldr, new Set(["🇯🇵", "🇺🇸", "🍰"]))
  expect(flags).toEqual([
    { text: "Japan", emojis: ["🇯🇵"], styles: ["Deadpan"], bg: ["#fffafa", "#d71920"], fg: "#17202a", src: "cldr" },
    { text: "United States", emojis: ["🇺🇸"], styles: ["Deadpan"], bg: ["#173f73", "#b22234"], fg: "#ffffff", src: "cldr" },
  ])
})

test("buildFlags drops flag emoji not in vocab and rows missing a palette", () => {
  const cldr = [
    { text: "Japan", emojis: "🇯🇵", styles: ["Deadpan"], ...P("#fffafa", "#d71920", "#17202a") },
    { text: "Chad", emojis: "🇹🇩", styles: ["Deadpan"] },
  ]
  expect(buildFlags(cldr, new Set(["🇹🇩"]))).toEqual([])
  expect(buildFlags(cldr, new Set(["🇯🇵"]))).toEqual([
    { text: "Japan", emojis: ["🇯🇵"], styles: ["Deadpan"], bg: ["#fffafa", "#d71920"], fg: "#17202a", src: "cldr" },
  ])
})
