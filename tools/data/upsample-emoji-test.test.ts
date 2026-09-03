import { expect, test } from "bun:test"

import { pickFailed } from "./upsample-emoji-test.ts"

test("pickFailed keeps words whose best rank is null or worse than maxRank", () => {
  const words = [
    { word: "coffee", expected: ["☕"], rank: 1 },
    { word: "bowl", expected: ["🥣"], rank: 8 },
    { word: "moon", expected: ["🌙"], rank: null },
    { word: "sun", expected: ["☀️"], rank: 5 },
  ]
  expect(pickFailed(words, 5)).toEqual([
    { word: "bowl", emoji: "🥣" },
    { word: "moon", emoji: "🌙" },
  ])
})

test("pickFailed targets the first expected emoji and skips words with none", () => {
  const words = [
    { word: "bowl", expected: ["🥣", "🍜"], rank: null },
    { word: "blank", expected: [], rank: null },
  ]
  expect(pickFailed(words, 5)).toEqual([{ word: "bowl", emoji: "🥣" }])
})
