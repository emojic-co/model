import { test, expect } from "bun:test"

import { countPalette } from "./upsample-emojis.ts"

test("countPalette counts only palette emojis, zero-fills the rest", () => {
  const rows = [
    { emoji: "📺" },
    { emoji: "📺" },
    { emoji: "🍕" },
    { emoji: "🛰️" },
  ]
  const counts = countPalette(rows, ["📺", "🍕", "🚗"])
  expect(counts.get("📺")).toBe(2)
  expect(counts.get("🍕")).toBe(1)
  expect(counts.get("🚗")).toBe(0)
  expect(counts.has("🛰️")).toBe(false)
})
