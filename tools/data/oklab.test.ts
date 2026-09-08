import { expect, test } from "bun:test"

import { chroma, hexToOklab, meanBgOklab } from "./oklab.ts"

test("hexToOklab maps black to L 0 and white to L 1, both achromatic", () => {
  const k = hexToOklab("#000000")
  const w = hexToOklab("#ffffff")
  expect(k[0]).toBeCloseTo(0, 5)
  expect(w[0]).toBeCloseTo(1, 3)
  expect(chroma(k)).toBeCloseTo(0, 4)
  expect(chroma(w)).toBeCloseTo(0, 4)
})

test("a saturated primary has far more chroma than a mid grey", () => {
  expect(chroma(hexToOklab("#00ff00"))).toBeGreaterThan(0.2)
  expect(chroma(hexToOklab("#808080"))).toBeLessThan(0.01)
})

test("meanBgOklab averages the two stops in oklab", () => {
  const m = meanBgOklab(["#000000", "#ffffff"])
  expect(m[0]).toBeCloseTo(0.5, 2)
})
