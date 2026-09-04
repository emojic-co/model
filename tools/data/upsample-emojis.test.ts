import { expect, test } from "bun:test"

import { countEmojis, failingEmojis, rankWindow } from "./upsample-emojis.ts"

test("countEmojis counts distinct vocab emojis per row, zero-fills the rest", () => {
  const rows = [
    { emojis: "📺 📺 🍕" },
    { emojis: "🍕 🚗" },
    { emojis: "🛰️" },
    {},
  ]
  const counts = countEmojis(rows, ["📺", "🍕", "🚗", "🎂"])
  expect(counts.get("📺")).toBe(1)
  expect(counts.get("🍕")).toBe(2)
  expect(counts.get("🚗")).toBe(1)
  expect(counts.get("🎂")).toBe(0)
  expect(counts.has("🛰️")).toBe(false)
})

test("rankWindow returns keys ranked [minRank, maxRank] by count desc, ties broken by vocab order", () => {
  const vocab = ["a", "b", "c", "d", "e"]
  const counts = new Map([
    ["a", 5],
    ["b", 0],
    ["c", 2],
    ["d", 0],
    ["e", 10],
  ])
  expect(rankWindow(vocab, counts, 1, 1)).toEqual(["e"])
  expect(rankWindow(vocab, counts, 2, 4)).toEqual(["a", "c", "b"])
  expect(rankWindow(vocab, counts, 4, 5)).toEqual(["b", "d"])
})

test("failingEmojis dedupes the first-expected emoji of words whose rank is null or worse than maxRank", () => {
  const words = [
    { keyword: "coffee", expected: ["☕"], rank: 1 },
    { keyword: "bowl", expected: ["🥣"], rank: 8 },
    { keyword: "soup", expected: ["🥣", "🍜"], rank: null },
    { keyword: "moon", expected: ["🌙"], rank: null },
    { keyword: "sun", expected: ["☀️"], rank: 5 },
    { keyword: "blank", expected: [], rank: null },
  ]
  expect(failingEmojis(words, 5)).toEqual(["🥣", "🌙"])
})
