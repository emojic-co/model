import { expect, test } from "bun:test"

import { pickColorRows } from "./extract-colors.ts"

test("pickColorRows keeps only rows with a non-empty string color field", () => {
  const rows = [
    { text: "a", emojis: "🧱", styles: ["Deadpan"], color: "red" },
    { text: "b", emojis: "", styles: ["Deadpan"] },
    { text: "c", emojis: "", styles: ["Deadpan"], color: "" },
    { text: "d", emojis: "", styles: ["Deadpan"], color: 3 },
    { text: "e", emojis: "", styles: ["Deadpan"], color: "blue" },
  ]
  expect(pickColorRows(rows as never).map((r) => r.text)).toEqual(["a", "e"])
})

test("pickColorRows passes rows through unchanged", () => {
  const row = {
    text: "Brick delivery is due at noon.",
    emojis: "🧱 🚚",
    styles: ["Deadpan"],
    bg: ["#d8b28a", "#bd936c"],
    fg: "#2b1b12",
    color: "red",
  }
  const out = pickColorRows([{ ...row }])
  expect(out).toHaveLength(1)
  expect(out[0]).toEqual(row)
})
