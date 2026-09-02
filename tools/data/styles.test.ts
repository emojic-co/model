import { expect, test } from "bun:test"

import { STYLE_LINES, STYLE_SET, STYLES } from "./styles.ts"

test("STYLES is a large unique list", () => {
  expect(STYLES.length).toBeGreaterThanOrEqual(20)
  expect(new Set(STYLES).size).toBe(STYLES.length)
})

test("every style has a non-empty annotator line", () => {
  for (const s of STYLES) {
    expect(typeof STYLE_LINES[s]).toBe("string")
    expect(STYLE_LINES[s].length).toBeGreaterThan(0)
  }
})

test("STYLE_SET matches STYLES", () => {
  expect(STYLE_SET.size).toBe(STYLES.length)
  for (const s of STYLES) expect(STYLE_SET.has(s)).toBe(true)
})
