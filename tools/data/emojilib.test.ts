import { expect, test } from "bun:test"

import { keywordText } from "./emojilib.ts"

test("keywordText replaces underscores with spaces and trims", () => {
  expect(keywordText("grinning_face")).toBe("grinning face")
  expect(keywordText("smile")).toBe("smile")
  expect(keywordText(" grin ")).toBe("grin")
  expect(keywordText(":D")).toBe(":D")
})
