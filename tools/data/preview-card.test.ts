import { expect, test } from "bun:test"

import { cardHtml } from "./preview-card.ts"

test("cardHtml accepts hex colors and renders a real gradient, not garbage from mis-typed OKLab math", () => {
  const html = cardHtml({
    text: "hello",
    emoji: "🙂",
    feeling: "Neutral",
    colors: { bg1: "#a8e2f4", bg2: "#78c9f4", text_color: "#282e36" },
  })
  expect(html).toContain("linear-gradient(135deg, #a8e2f4, #78c9f4)")
  expect(html).toContain("color: #282e36")
  expect(html).not.toContain("NaN")
  expect(html).not.toContain("undefined")
})
