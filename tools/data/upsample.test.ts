import { expect, test } from "bun:test"

import {
  colorBatchPlan,
  countEmojis,
  failingEmojis,
  missedCldrKeywords,
  parseKeywords,
  rankWindow,
  singleEmojiTexts,
} from "./upsample.ts"

test("parseKeywords trims, drops empties, and dedupes while preserving order", () => {
  expect(parseKeywords("rain, snow ,  , sunshine, rain")).toEqual([
    "rain",
    "snow",
    "sunshine",
  ])
  expect(parseKeywords("  ,, ")).toEqual([])
  expect(parseKeywords("dog walk")).toEqual(["dog walk"])
})

test("colorBatchPlan splits each colour's per-colour total into batches of at most batchSize", () => {
  const plan = colorBatchPlan(["red", "blue"], 100, 40)
  expect(plan).toEqual([
    { color: "red", n: 40 },
    { color: "red", n: 40 },
    { color: "red", n: 20 },
    { color: "blue", n: 40 },
    { color: "blue", n: 40 },
    { color: "blue", n: 20 },
  ])
})

test("colorBatchPlan emits one batch when per <= batchSize and sums each colour back to per", () => {
  const plan = colorBatchPlan(["red", "green", "blue"], 30, 50)
  expect(plan).toEqual([
    { color: "red", n: 30 },
    { color: "green", n: 30 },
    { color: "blue", n: 30 },
  ])
  const total = plan.filter((b) => b.color === "red").reduce((s, b) => s + b.n, 0)
  expect(total).toBe(30)
})

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

test("singleEmojiTexts keeps rows with a unique normalized text and at most one emoji", () => {
  const rows = [
    { text: "walking the dog", emojis: "🐕" },
    { text: "  Walking  the DOG  ", emojis: "🚶" },
    { text: "made pizza tonight", emojis: "🍕" },
    { text: "no emoji here", emojis: "" },
    { text: "two of them", emojis: "🍕 🚗" },
    { text: "same one twice", emojis: "😀 😀" },
    { text: "made pizza tonight", emojis: "🔥" },
  ]
  expect(singleEmojiTexts(rows, 100)).toEqual(["no emoji here", "same one twice"])
})

test("singleEmojiTexts: zero-emoji rows still need a unique normalized text", () => {
  const rows = [
    { text: "solo blank", emojis: "" },
    { text: "dup blank", emojis: "" },
    { text: "Dup  Blank", emojis: "🍕" },
  ]
  expect(singleEmojiTexts(rows, 100)).toEqual(["solo blank"])
})

test("singleEmojiTexts: collisions drop every colliding row, count caps in file order", () => {
  const rows = [
    { text: "alpha", emojis: "🍎" },
    { text: "beta", emojis: "🍌" },
    { text: "gamma", emojis: "🍇" },
    { text: "beta", emojis: "🐝 🐝 🍯" },
    { text: "delta", emojis: "🥝" },
  ]
  expect(singleEmojiTexts(rows, 2)).toEqual(["alpha", "gamma"])
  expect(singleEmojiTexts(rows, 100)).toEqual(["alpha", "gamma", "delta"])
})

test("singleEmojiTexts ignores rows with missing or non-string fields", () => {
  const rows = [
    { text: "keep me", emojis: "✅" },
    { emojis: "❌" },
    { text: "no emojis key" },
    { text: 5 as unknown as string, emojis: "🔢" },
    { text: "   ", emojis: "🌫️" },
  ]
  expect(singleEmojiTexts(rows, 100)).toEqual(["keep me"])
})

test("failingEmojis collects deduped targets from keywords not ranked within maxRank", () => {
  const misses = [
    { keyword: "bowl", targets: ["🥣"], rank: 8, top5: [], emoji_freq: 0, pair_freq: 0 },
    {
      keyword: "soup",
      targets: ["🥣", "🍜"],
      rank: 12,
      top5: [],
      emoji_freq: 0,
      pair_freq: 0,
    },
    { keyword: "moon", targets: ["🌙"], rank: 9, top5: [], emoji_freq: 0, pair_freq: 0 },
    { keyword: "sun", targets: ["☀️"], rank: 5, top5: [], emoji_freq: 0, pair_freq: 0 },
  ]
  expect(failingEmojis(misses, 5)).toEqual(["🥣", "🍜", "🌙"])
})

test("failingEmojis skips keywords whose pair_freq is at or above maxPairFreq", () => {
  const misses = [
    { keyword: "bowl", targets: ["🥣"], rank: 8, top5: [], emoji_freq: 0, pair_freq: 12 },
    {
      keyword: "soup",
      targets: ["🥣", "🍜"],
      rank: 12,
      top5: [],
      emoji_freq: 0,
      pair_freq: 50,
    },
    { keyword: "moon", targets: ["🌙"], rank: 9, top5: [], emoji_freq: 0, pair_freq: 99 },
  ]
  expect(failingEmojis(misses, 5, 50)).toEqual(["🥣"])
})

test("missedCldrKeywords dedupes keywords and honors maxPairFreq", () => {
  const misses = [
    { keyword: "rainy", targets: ["🌧️"], rank: null, top5: [], emoji_freq: 0, pair_freq: 3 },
    { keyword: "rainy", targets: ["🌧️", "☔"], rank: null, top5: [], emoji_freq: 0, pair_freq: 3 },
    { keyword: "party", targets: ["🎉"], rank: null, top5: [], emoji_freq: 0, pair_freq: 80 },
  ]
  expect(missedCldrKeywords(misses)).toEqual([
    { keyword: "rainy", targets: ["🌧️"] },
    { keyword: "party", targets: ["🎉"] },
  ])
  expect(missedCldrKeywords(misses, 50)).toEqual([
    { keyword: "rainy", targets: ["🌧️"] },
  ])
})
