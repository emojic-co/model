import { expect, test } from "bun:test"

import {
  colorBatchPlan,
  countEmojis,
  parseKeywords,
  rankWindow,
  rareEmojis,
  reannotateTexts,
  singleEmojiTexts,
  weightedMedian,
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

test("rareEmojis returns the rarest first, drops emoji below minFreq, ties by first-seen, capped at maxCount", () => {
  const counts = new Map([
    ["a", 5],
    ["b", 100],
    ["c", 20],
    ["d", 20],
    ["e", 3],
  ])
  expect(rareEmojis(counts, 10, 10)).toEqual(["c", "d", "b"])
  expect(rareEmojis(counts, 10, 2)).toEqual(["c", "d"])
  expect(rareEmojis(counts, 0, 3)).toEqual(["e", "a", "c"])
  expect(rareEmojis(counts, 200, 10)).toEqual([])
})

test("weightedMedian returns the lower median length of a [length, count] distribution", () => {
  expect(weightedMedian([[10, 1], [20, 1], [30, 1]])).toBe(20)
  expect(weightedMedian([[10, 2], [20, 2]])).toBe(10)
  expect(weightedMedian([[5, 3], [9, 1]])).toBe(5)
  expect(weightedMedian([[42, 7]])).toBe(42)
  expect(weightedMedian([[30, 5], [20, 5], [40, 1]])).toBe(30)
})

test("weightedMedian throws on an empty or zero-count distribution", () => {
  expect(() => weightedMedian([])).toThrow()
  expect(() => weightedMedian([[10, 0]])).toThrow()
})

test("singleEmojiTexts keeps rows with a unique normalized text and at most one emoji", () => {
  const rows = [
    { text: "walking the dog", emojis: "🐕" },
    { text: "  walking  the dog  ", emojis: "🚶" },
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
    { text: "dup  blank", emojis: "🍕" },
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

test("reannotateTexts samples rows with a palette, skips dups, palette-less, and already-done keys", () => {
  const rows = [
    { text: "alpha one", emojis: "🅰️", styles: ["Serene"], bg: ["#111111", "#222222"], fg: "#eeeeee" },
    { text: "beta two", emojis: "🅱️", styles: ["Tense"], bg: ["#333333", "#444444"], fg: "#dddddd" },
    { text: "  alpha  one ", emojis: "🔤", styles: [], bg: ["#555555", "#666666"], fg: "#cccccc" },
    { text: "gamma three", emojis: "", styles: ["Deadpan"], bg: ["#777777", "#888888"], fg: "#bbbbbb", reannotated: "colors" },
    { text: "gamma three", emojis: "", styles: ["Deadpan"], bg: ["#777777", "#888888"], fg: "#bbbbbb" },
    { text: "delta four", emojis: "🔺", styles: ["Playful"] },
    { text: "epsilon five", emojis: "5️⃣", styles: ["Wry"], bg: ["#999999", "#aaaaaa"], fg: "#000000" },
  ]
  const out = reannotateTexts(rows, 10)
  expect(out.map((r) => r.text).sort()).toEqual(["alpha one", "beta two", "epsilon five"])
  const alpha = out.find((r) => r.text === "alpha one")!
  expect(alpha.emojis).toBe("🅰️")
  expect(alpha.styles).toEqual(["Serene"])
  expect(alpha.bg).toEqual(["#111111", "#222222"])
  expect(alpha.fg).toBe("#eeeeee")
})

test("reannotateTexts caps at count after a deterministic seeded shuffle", () => {
  const rows = Array.from({ length: 20 }, (_, i) => ({
    text: `row number ${i}`,
    emojis: "",
    styles: ["Serene"],
    bg: ["#111111", "#222222"],
    fg: "#eeeeee",
  }))
  const a = reannotateTexts(rows, 5)
  const b = reannotateTexts(rows, 5)
  expect(a).toEqual(b)
  expect(a).toHaveLength(5)
  expect(new Set(a.map((r) => r.text)).size).toBe(5)
})
