import { expect, test } from "bun:test"

import { extractColorKeywords } from "./color-keywords.ts"

const P = (a: string, b: string, f: string) => ({ bg: [a, b], fg: f })

test("extractColorKeywords keeps only color-bearing records matching a keyword", () => {
  const out = extractColorKeywords(
    [
      { text: "dark chocolate cake", colors: [P("#000", "#111", "#fff")] },
      { text: "bright morning", colors: [P("#fff", "#eee", "#000")] },
      { text: "no colors here that mention dark", colors: undefined },
    ],
    ["dark", "chocolate"],
  )
  expect(out).toEqual([
    { keyword: "dark", text: "dark chocolate cake", colors: [P("#000", "#111", "#fff")] },
    { keyword: "chocolate", text: "dark chocolate cake", colors: [P("#000", "#111", "#fff")] },
  ])
})

test("extractColorKeywords matches case-insensitively and requires a boundary-padded substring", () => {
  const out = extractColorKeywords(
    [{ text: "Bred and Gold", colors: [P("#000", "#111", "#fff")] }],
    ["gold", " red "],
  )
  expect(out.map((r) => r.keyword)).toEqual(["gold"])
})
