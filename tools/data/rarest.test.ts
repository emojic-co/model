import { test, expect } from "bun:test"

import { rarest } from "./rarest.ts"

test("rarest returns lowest-count keys ascending", () => {
  const counts = new Map([["a", 5], ["b", 1], ["c", 3]])
  expect(rarest(["a", "b", "c"], counts, 2)).toEqual(["b", "c"])
})

test("rarest breaks ties by key order", () => {
  const counts = new Map([["a", 2], ["b", 2], ["c", 2]])
  expect(rarest(["c", "a", "b"], counts, 2)).toEqual(["c", "a"])
})

test("rarest treats missing keys as 0 and caps at keys.length", () => {
  const counts = new Map([["a", 4]])
  expect(rarest(["a", "b", "c"], counts, 10)).toEqual(["b", "c", "a"])
})
