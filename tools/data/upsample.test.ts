import { expect, test } from "bun:test"

import { countEmojis, failingEmojis, rankWindow } from "./upsample.ts"

test("countEmojis counts distinct emojis per row over all of data, not just a vocab", () => {
  const rows = [
    { emojis: "📺 📺 🍕" },
    { emojis: "🍕 🚗" },
    { emojis: "🛰️" },
    {},
  ]
  const counts = countEmojis(rows)
  expect(counts.get("📺")).toBe(1)
  expect(counts.get("🍕")).toBe(2)
  expect(counts.get("🚗")).toBe(1)
  expect(counts.get("🛰️")).toBe(1)
  expect(counts.has("🎂")).toBe(false)
})

test("rankWindow returns keys ranked [minRank, maxRank] by count desc, ties broken by first-seen order", () => {
  const counts = new Map([
    ["a", 5],
    ["b", 0],
    ["c", 2],
    ["d", 0],
    ["e", 10],
  ])
  expect(rankWindow(counts, 1, 1)).toEqual(["e"])
  expect(rankWindow(counts, 2, 4)).toEqual(["a", "c", "b"])
  expect(rankWindow(counts, 4, 5)).toEqual(["b", "d"])
})

test("failingEmojis dedupes target emoji worse than maxRank", () => {
  const misses = [
    { keyword: "bowl", target: "🥣", rank: 8, top5: [], emoji_freq: 0, pair_freq: 0 },
    { keyword: "soup", target: "🥣", rank: 12, top5: [], emoji_freq: 0, pair_freq: 0 },
    { keyword: "moon", target: "🌙", rank: 9, top5: [], emoji_freq: 0, pair_freq: 0 },
    { keyword: "sun", target: "☀️", rank: 5, top5: [], emoji_freq: 0, pair_freq: 0 },
  ]
  expect(failingEmojis(misses, 5)).toEqual(["🥣", "🌙"])
})
