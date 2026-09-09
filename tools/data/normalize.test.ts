import { expect, test } from "bun:test"

import { normalize } from "./normalize.ts"

test("collapses whitespace and trims, preserving case", () => {
  expect(normalize("  Hello    There  ")).toBe("Hello There")
})

test("collapses any run of 3+ identical chars to 2", () => {
  expect(normalize("Heyyy")).toBe("Heyy")
  expect(normalize("AAAA")).toBe("AA")
  expect(normalize("wow!!!")).toBe("wow!!")
})

test("keeps capital letters", () => {
  expect(normalize("NASA visited the UK")).toBe("NASA visited the UK")
})

test("drops characters outside the model vocab", () => {
  expect(normalize("don't stop")).toBe("dont stop")
  expect(normalize("1pm tea, please.")).toBe("1pm tea please")
  expect(normalize("hi 😀")).toBe("hi ")
})

test("keeps digits", () => {
  expect(normalize("call me at 0123456789")).toBe("call me at 0123456789")
})

test("keeps the punctuation the vocab allows", () => {
  expect(normalize("really?! (yes) @me & you")).toBe("really?! (yes) @me & you")
})
