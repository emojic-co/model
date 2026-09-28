import { expect, test } from "bun:test"

import { FontCache } from "./fonts.ts"
import { cardSvg } from "./svg-card.ts"

test("cardSvg accepts hex colors and renders a real gradient, not garbage from mis-typed OKLab math", () => {
  const svg = cardSvg(
    {
      text: "hello",
      emojis: "🙂",
      styles: ["Neutral"],
      colors: [{ bg: ["#a8e2f4", "#78c9f4"], fg: "#282e36" }],
    },
    512,
    new FontCache(),
  )
  expect(svg).toContain('stop-color="#a8e2f4"')
  expect(svg).toContain('stop-color="#78c9f4"')
  expect(svg).toContain('fill="#282e36"')
  expect(svg).not.toContain("NaN")
  expect(svg).not.toContain("undefined")
})
