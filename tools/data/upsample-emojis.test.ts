import { expect, test } from "bun:test"

import { countEmojis, rarest } from "./upsample-emojis.ts"

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

test("rarest returns the n lowest-count keys, ties broken by vocab order", () => {
  const vocab = ["a", "b", "c", "d"]
  const counts = new Map([
    ["a", 5],
    ["b", 0],
    ["c", 2],
    ["d", 0],
  ])
  expect(rarest(vocab, counts, 2)).toEqual(["b", "d"])
  expect(rarest(vocab, counts, 3)).toEqual(["b", "d", "c"])
})
