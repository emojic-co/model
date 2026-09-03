import { expect, test } from "bun:test"

import {
  colorSimilarity,
  deltaE,
  hexToRgb,
  mean,
  meanStd,
  oklab,
} from "./color-analysis.ts"

test("hexToRgb parses #rrggbb with or without the hash", () => {
  expect(hexToRgb("#ff8000")).toEqual([255, 128, 0])
  expect(hexToRgb("FF8000")).toEqual([255, 128, 0])
})

test("hexToRgb returns null for non-hex input", () => {
  expect(hexToRgb("#fff")).toBeNull()
  expect(hexToRgb("nope")).toBeNull()
  expect(hexToRgb("")).toBeNull()
})

test("oklab maps white to L~1 with ~0 chroma and black to the origin", () => {
  const [lw, aw, bw] = oklab([255, 255, 255])
  expect(lw).toBeCloseTo(1, 2)
  expect(aw).toBeCloseTo(0, 2)
  expect(bw).toBeCloseTo(0, 2)
  expect(oklab([0, 0, 0])).toEqual([0, 0, 0])
})

test("deltaE is zero for equal points and symmetric", () => {
  const p = oklab([12, 200, 90])
  const q = oklab([240, 30, 130])
  expect(deltaE(p, p)).toBe(0)
  expect(deltaE(p, q)).toBeCloseTo(deltaE(q, p), 12)
})

test("colorSimilarity is 100 for identical colors", () => {
  expect(colorSimilarity([192, 57, 43], [192, 57, 43])).toBe(100)
})

test("colorSimilarity clamps to 0 once distance exceeds the 0.5 span", () => {
  expect(colorSimilarity([0, 0, 0], [255, 255, 255])).toBe(0)
})

test("colorSimilarity is symmetric, within [0, 100], and rounded to 2dp", () => {
  const s = colorSimilarity([192, 57, 43], [210, 140, 150])
  expect(s).toBe(colorSimilarity([210, 140, 150], [192, 57, 43]))
  expect(s).toBeGreaterThan(0)
  expect(s).toBeLessThan(100)
  expect(Number(s.toFixed(2))).toBe(s)
})

test("colorSimilarity ranks a nearer color higher than a farther one", () => {
  const red = [192, 57, 43]
  const nearRed = [200, 80, 70]
  const blue = [74, 111, 209]
  expect(colorSimilarity(red, nearRed)).toBeGreaterThan(
    colorSimilarity(red, blue),
  )
})

test("mean and meanStd over a known sample", () => {
  expect(mean([2, 4, 9])).toBeCloseTo(5, 12)
  const { mean: m, std } = meanStd([2, 4, 4, 4, 5, 5, 7, 9])
  expect(m).toBeCloseTo(5, 12)
  expect(std).toBeCloseTo(2, 12)
})

test("mean and meanStd are safe on an empty list", () => {
  expect(mean([])).toBe(0)
  expect(meanStd([])).toEqual({ mean: 0, std: 0 })
})
