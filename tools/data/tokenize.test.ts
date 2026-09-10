import { expect, test } from "bun:test"

import { queryTokens, STOPWORDS } from "./tokenize.ts"

test("splits, lowercases, strips punctuation", () => {
  expect(queryTokens("The Dog is Running-FAST, y'all!")).toEqual(["dog", "running", "fast", "all"])
})

test("drops stopwords and 1-char tokens", () => {
  expect(STOPWORDS.has("the")).toBe(true)
  expect(queryTokens("a cat and the hat i see")).toEqual(["cat", "hat", "see"])
})

test("keeps digits", () => {
  expect(queryTokens("bus 42 now")).toEqual(["bus", "42", "now"])
})
