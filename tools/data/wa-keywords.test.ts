import { expect, test } from "bun:test"

import { toSourceRows } from "./wa-keywords.ts"

test("toSourceRows maps each entry to a styleless SourceRow with space-joined emoji", () => {
  expect(toSourceRows({ cake: ["🍰", "🎂"], rocket: ["🚀"] })).toEqual([
    { text: "cake", emojis: "🍰 🎂", styles: [] },
    { text: "rocket", emojis: "🚀", styles: [] },
  ])
})
