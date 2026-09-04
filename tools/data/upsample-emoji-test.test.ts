import { expect, test } from "bun:test"

import { pickFailed } from "./upsample-emoji-test.ts"

test("pickFailed keeps words whose best rank is null or worse than maxRank", () => {
  const words = [
    { keyword: "coffee", expected: ["☕"], rank: 1 },
    { keyword: "bowl", expected: ["🥣"], rank: 8 },
    { keyword: "moon", expected: ["🌙"], rank: null },
    { keyword: "sun", expected: ["☀️"], rank: 5 },
  ]
  expect(pickFailed(words, 5)).toEqual([
    { word: "bowl", emoji: "🥣" },
    { word: "moon", emoji: "🌙" },
  ])
})

test("pickFailed targets the first expected emoji and skips words with none", () => {
  const words = [
    { keyword: "bowl", expected: ["🥣", "🍜"], rank: null },
    { keyword: "blank", expected: [], rank: null },
  ]
  expect(pickFailed(words, 5)).toEqual([{ word: "bowl", emoji: "🥣" }])
})
