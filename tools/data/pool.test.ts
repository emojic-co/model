import { expect, test } from "bun:test"

import { dedupe, freqBucket, sortPool, stableHash } from "./pool.ts"

test("dedupe drops normalized duplicates, keeping the first", () => {
  const { kept, duplicate, degenerate } = dedupe([
    { text: "Hello there", id: 1 },
    { text: "  hello    there  ", id: 2 },
    { text: "Heyyy", id: 3 },
    { text: "heyy", id: 4 },
    { text: "different", id: 5 },
  ])
  expect(kept.map((r) => r.id)).toEqual([1, 3, 5])
  expect(duplicate).toBe(2)
  expect(degenerate).toBe(0)
})

test("dedupe drops rows that normalize to empty or over the length cap", () => {
  const long = "the quick brown fox jumps over the lazy dog and then some"
  const { kept, degenerate } = dedupe([
    { text: "12345" },
    { text: "ok text" },
    { text: long },
  ])
  expect(kept.map((r) => r.text)).toEqual(["ok text"])
  expect(degenerate).toBe(2)
})

test("stableHash is deterministic and unsigned", () => {
  expect(stableHash("abc")).toBe(stableHash("abc"))
  expect(stableHash("abc")).not.toBe(stableHash("abd"))
  expect(stableHash("whatever string")).toBeGreaterThanOrEqual(0)
})

test("freqBucket is monotone non-decreasing in count", () => {
  expect(freqBucket(0)).toBe(0)
  expect(freqBucket(1)).toBe(1)
  expect(freqBucket(3)).toBe(2)
  expect(freqBucket(1000)).toBeGreaterThan(freqBucket(10))
})

test("sortPool puts non-face rows before face rows and is deterministic", () => {
  const rows = [
    { text: "a train is late", _emoji: "😤" },
    { text: "fresh bread today", _emoji: "🥖" },
    { text: "another angry one", _emoji: "😠" },
    { text: "the bus arrived", _emoji: "🚌" },
  ]
  const a = sortPool([...rows]).map((r) => r.text)
  expect(sortPool([...rows]).map((r) => r.text)).toEqual(a)

  const faceTexts = new Set(["a train is late", "another angry one"])
  const faceIdx = a.map((t, i) => (faceTexts.has(t) ? i : -1)).filter((i) => i >= 0)
  const nonFaceIdx = a
    .map((t, i) => (faceTexts.has(t) ? -1 : i))
    .filter((i) => i >= 0)
  expect(Math.max(...nonFaceIdx)).toBeLessThan(Math.min(...faceIdx))
})
