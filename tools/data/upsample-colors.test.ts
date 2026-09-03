import { expect, test } from "bun:test"

import { parseColors } from "./upsample-colors.ts"

const LIST = ["red", "green", "blue", "black"]

test("parseColors returns the whole list (a copy) when no --colors flag", () => {
  const out = parseColors(["bun", "upsample-colors.ts"], LIST)
  expect(out).toEqual(LIST)
  expect(out).not.toBe(LIST)
})

test("parseColors returns the named subset in argv order", () => {
  expect(parseColors(["x", "--colors", "blue", "red"], LIST)).toEqual([
    "blue",
    "red",
  ])
})

test("parseColors lowercases and de-dupes, keeping first-seen order", () => {
  expect(parseColors(["x", "--colors", "RED", "red", "Blue"], LIST)).toEqual([
    "red",
    "blue",
  ])
})

test("parseColors stops collecting at the next flag", () => {
  expect(parseColors(["x", "--colors", "green", "--per", "3"], LIST)).toEqual([
    "green",
  ])
})

test("parseColors throws on an unknown color", () => {
  expect(() => parseColors(["x", "--colors", "red", "teal"], LIST)).toThrow(
    /teal/,
  )
})

test("parseColors throws when --colors has no names after it", () => {
  expect(() => parseColors(["x", "--colors"], LIST)).toThrow()
})
