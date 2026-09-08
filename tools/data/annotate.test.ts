import { expect, test } from "bun:test"

import {
  FALLBACK_PALETTES,
  MIN_CONTRAST,
  PALETTE_GUIDANCE,
  contrast,
  paletteInstructions,
  repairFg,
  resolvePalette,
  resolvePaletteRepairing,
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

test("PALETTE_GUIDANCE steers toward saturation and expressive lightness", () => {
  const s = PALETTE_GUIDANCE.join(" ").toLowerCase()
  expect(s).toContain("saturated")
  expect(s).toContain("pastels")
  expect(s).toContain("dark")
  expect(s).toContain("bright")
})

test("paletteInstructions asks for a palette only, no emoji or style rules", () => {
  const s = paletteInstructions()
  expect(s).toContain("#rrggbb")
  expect(s).toMatch(/\bbg\b/)
  expect(s).toMatch(/\bfg\b/)
  expect(s).toContain(PALETTE_GUIDANCE[0])
  expect(s.toLowerCase()).not.toContain("emoji")
  expect(s.toLowerCase()).not.toContain("styles")
})

test("repairFg leaves an already-readable fg untouched", () => {
  expect(repairFg(["#eef2f6", "#dbe3ec"], "#26323f")).toBe("#26323f")
})

test("repairFg darkens a too-light fg on a mid-value bg until both stops clear MIN_CONTRAST", () => {
  const bg: [string, string] = ["#f28c28", "#e84d3c"]
  const fixed = repairFg(bg, "#fff8e8")
  expect(fixed).not.toBeNull()
  expect(contrast(fixed!, bg[0])).toBeGreaterThanOrEqual(MIN_CONTRAST)
  expect(contrast(fixed!, bg[1])).toBeGreaterThanOrEqual(MIN_CONTRAST)
})

test("repairFg lightens a too-dark fg on a mid-value bg", () => {
  const bg: [string, string] = ["#3f9b9a", "#2f687f"]
  const fixed = repairFg(bg, "#102f3b")
  expect(fixed).not.toBeNull()
  expect(contrast(fixed!, bg[0])).toBeGreaterThanOrEqual(MIN_CONTRAST)
  expect(contrast(fixed!, bg[1])).toBeGreaterThanOrEqual(MIN_CONTRAST)
})

test("repairFg gives up when the gradient spans too much lightness for any fg", () => {
  expect(repairFg(["#1a1a1a", "#b0b0b0"], "#808080")).toBeNull()
})

test("resolvePaletteRepairing flags malformed hex", () => {
  const r = resolvePaletteRepairing(["nope", "#111111"], "#eeeeee")
  expect(r.ok).toBe(false)
  if (!r.ok) expect(r.reason).toBe("badHex")
})

test("resolvePaletteRepairing repairs a low-contrast fg and marks it repaired", () => {
  const r = resolvePaletteRepairing(["#f28c28", "#e84d3c"], "#fff8e8")
  expect(r.ok).toBe(true)
  if (r.ok) {
    expect(r.repaired).toBe(true)
    expect(r.bg).toEqual(["#f28c28", "#e84d3c"])
    expect(contrast(r.fg, r.bg[0])).toBeGreaterThanOrEqual(MIN_CONTRAST)
    expect(contrast(r.fg, r.bg[1])).toBeGreaterThanOrEqual(MIN_CONTRAST)
  }
})

test("resolvePaletteRepairing passes a good palette through unrepaired", () => {
  const r = resolvePaletteRepairing(["#eef2f6", "#dbe3ec"], "#26323f")
  expect(r.ok).toBe(true)
  if (r.ok) {
    expect(r.repaired).toBe(false)
    expect(r.fg).toBe("#26323f")
  }
})
