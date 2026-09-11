import { expect, test } from "bun:test"

import { accAtK, hitAtK, makeSearch, rankByScore, rowTargets } from "./kw-search.ts"

test("makeSearch scores an exact keyword hit with its idf weight plus PRIMARY_BONUS", () => {
  const search = makeSearch({ pizza: [0, 1] })
  expect(search.predict("i love pizza")).toEqual([
    [0, 0.781],
    [1, 0.631],
  ])
})

test("makeSearch adds PRIMARY_BONUS to the first (popularity-sorted) posting only", () => {
  const search = makeSearch({ dog: [2, 5] })
  const kw = new Map(search.predict("a dog"))
  expect(kw.get(2)! - kw.get(5)!).toBeCloseTo(0.15)
})

test("makeSearch fuzzy-matches a truncated keyword below exact strength", () => {
  const search = makeSearch({ pizza: [0] })
  const exact = new Map(search.predict("pizza")).get(0)!
  const fuzzy = new Map(search.predict("pizz")).get(0)!
  expect(fuzzy).toBeGreaterThan(0)
  expect(fuzzy).toBeLessThan(exact)
})

test("makeSearch ignores stopwords and short tokens", () => {
  const search = makeSearch({ it: [0], a: [1] })
  expect(search.predict("it is a test")).toEqual([])
})

test("rankByScore sorts descending by score, ties broken by ascending index", () => {
  expect(
    rankByScore(
      [
        [0, 0.5],
        [1, 0.9],
        [2, 0.5],
      ],
      3,
    ),
  ).toEqual([1, 0, 2])
})

test("hitAtK checks whether any target lands in the top k", () => {
  const ranked = [3, 1, 4, 0]
  expect(hitAtK(ranked, [4], 2)).toBe(false)
  expect(hitAtK(ranked, [4], 3)).toBe(true)
  expect(hitAtK(ranked, [9], 10)).toBe(false)
})

test("rowTargets splits on whitespace, dedupes, drops out-of-vocab glyphs", () => {
  const idx = new Map([
    ["🍕", 0],
    ["🧀", 1],
  ])
  expect(rowTargets("🍕 🧀 🍕 🦄", idx)).toEqual([0, 1])
})

test("accAtK computes acc@k over the requested k values", () => {
  const rows = [
    { text: "a", targets: [0] },
    { text: "b", targets: [1] },
  ]
  const scored = [
    { text: "a", kw: [[0, 1] as [number, number]] },
    { text: "b", kw: [[2, 1] as [number, number]] },
  ]
  const acc = accAtK(rows, scored, 3, [1, 2])
  expect(acc[1]).toBeCloseTo(0.5)
  expect(acc[2]).toBeCloseTo(0.5)
})
