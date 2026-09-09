import { expect, test } from "bun:test"

import { computeRemojiStats } from "./remojis.ts"

const rows = [
  { emojis: "🅰️ 🎉 🍰", remojis: ["🎉", "🍰"], meta: { date: "2026-09-09" } },
  { emojis: "🅱️", remojis: [], meta: { date: "2026-09-09" } },
  { emojis: "🚗 🛞", remojis: ["🛞"], meta: { date: "2026-09-10" } },
  { emojis: "🎈", remojis: [], meta: {} },
  { emojis: "no remojis field here" },
]

test("computeRemojiStats only counts rows with a remojis array", () => {
  const s = computeRemojiStats(rows)
  expect(s.records).toBe(4)
})

test("computeRemojiStats totals and means", () => {
  const s = computeRemojiStats(rows)
  expect(s.totalAdded).toBe(3)
  expect(s.withAdditions).toBe(2)
  expect(s.zeroAdditions).toBe(2)
  expect(s.meanAddedPerRecord).toBeCloseTo(0.75)
  expect(s.meanAddedPerExpanded).toBeCloseTo(1.5)
  expect(s.distinctAdded).toBe(3)
})

test("computeRemojiStats derives original vs final emoji counts", () => {
  const s = computeRemojiStats(rows)
  expect(s.meanOrigCount).toBeCloseTo((1 + 1 + 1 + 1) / 4)
  expect(s.meanFinalCount).toBeCloseTo((3 + 1 + 2 + 1) / 4)
  expect(s.expandedExisting).toBe(2)
  expect(s.expandedFromEmpty).toBe(0)
})

test("computeRemojiStats histograms and top list", () => {
  const s = computeRemojiStats(rows)
  expect(s.addedCountHistogram).toEqual([
    { added: 0, records: 2 },
    { added: 1, records: 1 },
    { added: 2, records: 1 },
  ])
  expect(s.topAdded[0]).toMatchObject({ count: 1 })
  expect(s.topAdded).toHaveLength(3)
})

test("computeRemojiStats groups by meta.date", () => {
  const s = computeRemojiStats(rows)
  expect(s.byDate).toEqual([
    { date: "(none)", records: 1, totalAdded: 0, meanAdded: 0 },
    { date: "2026-09-09", records: 2, totalAdded: 2, meanAdded: 1 },
    { date: "2026-09-10", records: 1, totalAdded: 1, meanAdded: 1 },
  ])
})
