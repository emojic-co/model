import { expect, test } from "bun:test"

import { normalize } from "./normalize.ts"

test("collapses whitespace, trims, lowercases", () => {
  expect(normalize("  Hello    There  ")).toBe("hello there")
})

test("collapses any run of 3+ identical chars to 2", () => {
  expect(normalize("Heyyy")).toBe("heyy")
  expect(normalize("AAAA")).toBe("aa")
  expect(normalize("wow!!!")).toBe("wow!!")
})

test("drops characters outside the model vocab", () => {
  expect(normalize("don't stop")).toBe("dont stop")
  expect(normalize("1pm tea, please.")).toBe("pm tea please")
  expect(normalize("hi 😀")).toBe("hi ")
})

test("keeps the punctuation the vocab allows", () => {
  expect(normalize("really?! (yes) @me & you")).toBe("really?! (yes) @me & you")
})
