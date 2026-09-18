import { expect, test } from "bun:test"

import { buildColorTerms, derivePalette, hexToHsl, hslToHex } from "./color-names.ts"

test("hslToHex converts pure hues correctly", () => {
  expect(hslToHex({ h: 0, s: 1, l: 0.5 })).toBe("#ff0000")
  expect(hslToHex({ h: 120, s: 1, l: 0.5 })).toBe("#00ff00")
  expect(hslToHex({ h: 240, s: 1, l: 0.5 })).toBe("#0000ff")
  expect(hslToHex({ h: 0, s: 0, l: 0 })).toBe("#000000")
  expect(hslToHex({ h: 0, s: 0, l: 1 })).toBe("#ffffff")
})

test("hexToHsl round-trips through hslToHex", () => {
  for (const hex of ["#5d8aa8", "#e32636", "#ffbf00", "#000000", "#ffffff", "#808080"]) {
    expect(hslToHex(hexToHsl(hex))).toBe(hex)
  }
})

function hexLuma(hex: string): number {
  const n = parseInt(hex.replace(/^#/, ""), 16)
  return ((n >> 16) & 255) + ((n >> 8) & 255) + (n & 255)
}

test("derivePalette keeps the real hex as bg1 and picks a contrasting fg", () => {
  const dark = derivePalette("#26130f")
  expect(dark.bg[0]).toBe("#26130f")
  expect(hexLuma(dark.fg)).toBeGreaterThan(hexLuma(dark.bg[0]))
  expect(hexLuma(dark.fg)).toBeGreaterThan(hexLuma(dark.bg[1]))

  const light = derivePalette("#f0f8ff")
  expect(light.bg[0]).toBe("#f0f8ff")
  expect(hexLuma(light.fg)).toBeLessThan(hexLuma(light.bg[0]))
  expect(hexLuma(light.fg)).toBeLessThan(hexLuma(light.bg[1]))
})

test("buildColorTerms lowercases names, dedupes, and keeps valid hex fields", () => {
  const terms = buildColorTerms([
    { name: "Air Force blue", hex: "#5d8aa8" },
    { name: "Alice blue", hex: "#f0f8ff" },
    { name: "alice blue", hex: "#f0f8ff" },
    { name: "Dark blue", hex: "#00008b" },
    { name: "Light blue", hex: "#add8e6" },
    { name: "Pale green", hex: "#98fb98" },
  ])
  expect(terms.length).toBe(5)

  const hex = /^#[0-9a-f]{6}$/
  for (const t of terms) {
    expect(t.bg[0]).toMatch(hex)
    expect(t.bg[1]).toMatch(hex)
    expect(t.fg).toMatch(hex)
  }

  const byText = Object.fromEntries(terms.map((t) => [t.text, t]))
  expect(byText["air force blue"].bg[0]).toBe("#5d8aa8")
  expect(byText["dark blue"]).toBeDefined()
  expect(byText["light blue"]).toBeDefined()
  expect(byText["pale green"]).toBeDefined()
})
