import { test, expect } from "bun:test"

import { countPaletteFeelings } from "./upsample-feelings.ts"

test("countPaletteFeelings counts rows in-palette on both axes", () => {
  const rows = [
    { emoji: "📺", feeling: "Calm" },
    { emoji: "📺", feeling: "Calm" },
    { emoji: "🛰️", feeling: "Calm" },
    { emoji: "📺", feeling: "Rapturous" },
  ]
  const counts = countPaletteFeelings(rows, ["Calm", "Sad"], new Set(["📺"]))
  expect(counts.get("Calm")).toBe(2)
  expect(counts.get("Sad")).toBe(0)
  expect(counts.has("Rapturous")).toBe(false)
})
