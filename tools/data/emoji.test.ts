import { expect, test } from "bun:test"

import { coarseEmojiGroup, isFaceEmoji, splitEmojis } from "./emoji.ts"

test("splitEmojis splits on whitespace and keeps ZWJ / VS16 sequences intact", () => {
  expect(splitEmojis("🥰 🐶")).toEqual(["🥰", "🐶"])
  expect(splitEmojis("  🚌   😤 ")).toEqual(["🚌", "😤"])
  expect(splitEmojis("😮‍💨")).toEqual(["😮‍💨"])
  expect(splitEmojis("🍽️")).toEqual(["🍽️"])
})

test("splitEmojis drops empty input and plain ascii tokens", () => {
  expect(splitEmojis("")).toEqual([])
  expect(splitEmojis("   ")).toEqual([])
  expect(splitEmojis("hello 🐶")).toEqual(["🐶"])
})

test("splitEmojis recovers when the model omits the separator", () => {
  expect(splitEmojis("🥰🐶")).toEqual(["🥰", "🐶"])
})

test("isFaceEmoji flags smileys, not objects or animals", () => {
  for (const e of ["😤", "😠", "🤢", "🥰", "🫠", "☹"]) {
    expect(isFaceEmoji(e)).toBe(true)
  }
  for (const e of ["🚌", "🐶", "☕", "📦", "🌧️", "🎉"]) {
    expect(isFaceEmoji(e)).toBe(false)
  }
})

test("coarseEmojiGroup lands common emojis in a plausible group", () => {
  expect(coarseEmojiGroup("😤")).toBe("Smileys & Emotion")
  expect(coarseEmojiGroup("🐶")).toBe("Animals & Nature")
  expect(coarseEmojiGroup("🍕")).toBe("Food & Drink")
  expect(coarseEmojiGroup("🚌")).toBe("Travel & Places")
})
