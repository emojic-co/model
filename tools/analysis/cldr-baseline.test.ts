import { expect, test } from "bun:test"

import {
  hitAtK,
  rankPredictions,
  reciprocalRank,
  rowTargets,
  stripVS,
  summarize,
} from "./cldr-baseline.ts"

test("stripVS removes emoji variation selectors, leaves plain glyphs untouched", () => {
  expect(stripVS("⏱️")).toBe("⏱")
  expect(stripVS("⚙️")).toBe("⚙")
  expect(stripVS("🍕")).toBe("🍕")
})

test("rowTargets splits on whitespace, strips selectors, keeps vocab hits, dedupes", () => {
  const vocab = new Set(["⏱", "⚙", "🍕"])
  expect(rowTargets("⏱️ ⚙️ 🍕 ⏱️", vocab)).toEqual(["⏱", "⚙", "🍕"])
})

test("rowTargets returns empty when no emoji is in the vocab", () => {
  expect(rowTargets("🦄 🐉", new Set(["🍕"]))).toEqual([])
})

test("rankPredictions strips selectors, filters to vocab, dedupes preserving order", () => {
  const vocab = new Set(["🍕", "🧀"])
  expect(rankPredictions(["🍕️", "🦄", "🧀", "🍕"], vocab)).toEqual(["🍕", "🧀"])
})

test("hitAtK is true only when a target appears within the first k predictions", () => {
  const preds = ["🦄", "🍕", "🧀"]
  expect(hitAtK(preds, ["🍕"], 1)).toBe(false)
  expect(hitAtK(preds, ["🍕"], 2)).toBe(true)
  expect(hitAtK(preds, ["🐉"], 10)).toBe(false)
})

test("reciprocalRank returns 1/rank of the first relevant prediction, else 0", () => {
  expect(reciprocalRank(["🦄", "🍕", "🧀"], ["🧀", "🍕"])).toBeCloseTo(1 / 2)
  expect(reciprocalRank(["🦄", "🐉"], ["🍕"])).toBe(0)
})

test("summarize aggregates acc@k (k=1..10), MRR and prediction coverage over rows", () => {
  const s = summarize([
    { preds: ["🍕", "🧀"], targets: ["🍕"] },
    { preds: ["🦄", "🐉", "🧀"], targets: ["🧀"] },
    { preds: [], targets: ["⚙"] },
  ])
  expect(s.n).toBe(3)
  expect(s.accAtK).toHaveLength(10)
  expect(s.accAtK[0]).toBeCloseTo(1 / 3)
  expect(s.accAtK[2]).toBeCloseTo(2 / 3)
  expect(s.mrr).toBeCloseTo((1 + 1 / 3 + 0) / 3)
  expect(s.meanPreds).toBeCloseTo((2 + 3 + 0) / 3)
  expect(s.zeroPredRows).toBe(1)
})
