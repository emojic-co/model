import { expect, test } from "bun:test"

import { invertedIndex, missingRecords, stripTtsPrefix } from "./cldr.ts"

test("stripTtsPrefix drops a leading `word: ` label, keeps the rest verbatim", () => {
  expect(stripTtsPrefix("flag: Japan")).toBe("Japan")
  expect(stripTtsPrefix("flag: St. Kitts & Nevis")).toBe("St. Kitts & Nevis")
  expect(stripTtsPrefix("keycap: 10")).toBe("10")
  expect(stripTtsPrefix("grinning face")).toBe("grinning face")
  expect(stripTtsPrefix("family: man, woman, boy")).toBe("man, woman, boy")
})

test("invertedIndex maps keyword -> sorted deduped emoji list, records sorted by keyword", () => {
  const annotations = new Map([
    ["🍕", ["pizza", "slice", "cheese"]],
    ["🧀", ["cheese", "dairy"]],
    ["🍰", ["cake", "slice"]],
  ])
  expect(invertedIndex(annotations, 10)).toEqual([
    { text: "cake", emojis: ["🍰"] },
    { text: "cheese", emojis: ["🍕", "🧀"] },
    { text: "dairy", emojis: ["🧀"] },
    { text: "pizza", emojis: ["🍕"] },
    { text: "slice", emojis: ["🍕", "🍰"] },
  ])
})

test("invertedIndex drops keywords shorter than 2 chars after trimming", () => {
  const annotations = new Map([
    ["🅰️", ["a", " b ", "ok", "  ", "x"]],
  ])
  expect(invertedIndex(annotations, 10)).toEqual([{ text: "ok", emojis: ["🅰️"] }])
})

test("invertedIndex drops over-generic buckets with more than maxEmojis emojis", () => {
  const annotations = new Map([
    ["😀", ["face", "smile"]],
    ["😃", ["face", "smile"]],
    ["😄", ["face"]],
    ["😁", ["face"]],
  ])
  expect(invertedIndex(annotations, 3)).toEqual([{ text: "smile", emojis: ["😀", "😃"] }])
  expect(invertedIndex(annotations, 4).map((r) => r.text)).toEqual(["face", "smile"])
})

test("invertedIndex dedupes a keyword repeated on the same emoji", () => {
  const annotations = new Map([["🐕", ["dog", "dog", "pet"]]])
  expect(invertedIndex(annotations, 10)).toEqual([
    { text: "dog", emojis: ["🐕"] },
    { text: "pet", emojis: ["🐕"] },
  ])
})

test("missingRecords keeps only records whose text is not already present", () => {
  const records = [
    { text: "cake", emojis: ["🍰"] },
    { text: "cheese", emojis: ["🍕", "🧀"] },
    { text: "pizza", emojis: ["🍕"] },
  ]
  expect(missingRecords(records, new Set(["cheese"]))).toEqual([
    { text: "cake", emojis: ["🍰"] },
    { text: "pizza", emojis: ["🍕"] },
  ])
  expect(missingRecords(records, new Set())).toEqual(records)
  expect(missingRecords(records, new Set(["cake", "cheese", "pizza"]))).toEqual([])
})
