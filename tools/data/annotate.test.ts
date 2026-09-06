import { expect, test } from "bun:test"

import {
  FALLBACK_PALETTES,
  MIN_CONTRAST,
  contrast,
  resolvePalette,
} from "./annotate.ts"

test("resolvePalette passes a valid annotator palette straight through", () => {
  const r = resolvePalette(["#c9d8e5", "#9fb4c8"], "#172b3a", false)
  expect(r.filled).toBe(false)
  expect(r.palette).toEqual({ bg: ["#c9d8e5", "#9fb4c8"], fg: "#172b3a" })
})

test("resolvePalette returns no palette for junk colors when fill is off", () => {
  expect(resolvePalette(undefined, undefined, false)).toEqual({
    palette: null,
    filled: false,
  })
  expect(resolvePalette(["#000000", "#000000"], "#010101", false)).toEqual({
    palette: null,
    filled: false,
  })
})

test("resolvePalette fills from the fallback set for junk colors when fill is on", () => {
  for (let i = 0; i < 50; i++) {
    const r = resolvePalette(["nope"], "", true)
    expect(r.filled).toBe(true)
    expect(FALLBACK_PALETTES).toContainEqual({
      bg: r.palette?.bg,
      fg: r.palette?.fg,
    })
  }
})

test("there are exactly 10 fallback palettes and every one is readable", () => {
  expect(FALLBACK_PALETTES).toHaveLength(10)
  for (const p of FALLBACK_PALETTES) {
    expect(contrast(p.fg, p.bg[0])).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(contrast(p.fg, p.bg[1])).toBeGreaterThanOrEqual(MIN_CONTRAST)
  }
})
