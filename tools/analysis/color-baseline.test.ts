import { expect, test } from "bun:test"

import {
  deltaE,
  hexToRgb,
  meanVec,
  mse,
  paletteToOklab,
  predictStyleMean,
  srgbByteToOklab,
  styleMeans,
  summarize,
} from "./color-baseline.ts"

test("hexToRgb parses a 6-digit hex with or without leading #", () => {
  expect(hexToRgb("#ff0000")).toEqual([255, 0, 0])
  expect(hexToRgb("00ff80")).toEqual([0, 255, 128])
})

test("srgbByteToOklab maps white to L=1, a=0, b=0", () => {
  const [l, a, b] = srgbByteToOklab([255, 255, 255])
  expect(l).toBeCloseTo(1, 3)
  expect(a).toBeCloseTo(0, 3)
  expect(b).toBeCloseTo(0, 3)
})

test("srgbByteToOklab maps black to L=0, a=0, b=0", () => {
  const [l, a, b] = srgbByteToOklab([0, 0, 0])
  expect(l).toBeCloseTo(0, 3)
  expect(a).toBeCloseTo(0, 3)
  expect(b).toBeCloseTo(0, 3)
})

test("srgbByteToOklab maps mid grey to a neutral chroma with L about 0.6", () => {
  const [l, a, b] = srgbByteToOklab([128, 128, 128])
  expect(l).toBeCloseTo(0.6, 2)
  expect(a).toBeCloseTo(0, 3)
  expect(b).toBeCloseTo(0, 3)
})

test("paletteToOklab returns a 9-vector: bg1, bg2, fg each as an OKLab triple", () => {
  const v = paletteToOklab(["#ffffff", "#000000"], "#ffffff")
  expect(v).toHaveLength(9)
  expect(v[0]).toBeCloseTo(1, 3)
  expect(v[3]).toBeCloseTo(0, 3)
  expect(v[6]).toBeCloseTo(1, 3)
})

test("meanVec averages element-wise across vectors", () => {
  expect(meanVec([[0, 2, 4], [2, 4, 8]])).toEqual([1, 3, 6])
})

test("deltaE is the mean L2 distance over the three colour triples", () => {
  const a = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  const b = [3, 4, 0, 0, 0, 0, 0, 0, 0]
  expect(deltaE(a, b)).toBeCloseTo(5 / 3)
})

test("mse is the mean squared error over all nine components", () => {
  const a = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  const b = [3, 0, 0, 0, 0, 0, 0, 0, 0]
  expect(mse(a, b)).toBeCloseTo(9 / 9)
})

test("styleMeans averages the OKLab palette per style label", () => {
  const means = styleMeans([
    { styles: ["Calm"], oklab: [0, 0, 0, 0, 0, 0, 0, 0, 0] },
    { styles: ["Calm", "Tense"], oklab: [2, 0, 0, 0, 0, 0, 0, 0, 0] },
    { styles: ["Tense"], oklab: [4, 0, 0, 0, 0, 0, 0, 0, 0] },
  ])
  expect(means.get("Calm")![0]).toBeCloseTo(1)
  expect(means.get("Tense")![0]).toBeCloseTo(3)
})

test("predictStyleMean averages the row's style means, falling back to global for unseen styles", () => {
  const means = new Map([
    ["Calm", [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    ["Tense", [4, 0, 0, 0, 0, 0, 0, 0, 0]],
  ])
  const global = [9, 0, 0, 0, 0, 0, 0, 0, 0]
  expect(predictStyleMean(["Calm", "Tense"], means, global)[0]).toBeCloseTo(2)
  expect(predictStyleMean(["Unknown"], means, global)[0]).toBeCloseTo(9)
  expect(predictStyleMean([], means, global)[0]).toBeCloseTo(9)
})

test("summarize reports mean deltaE and mse of predictions against targets", () => {
  const preds = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
  const targets = [[3, 4, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]
  const s = summarize(preds, targets)
  expect(s.dE).toBeCloseTo((5 / 3 + 0) / 2)
  expect(s.mse).toBeCloseTo((25 / 9 + 0) / 2)
})
