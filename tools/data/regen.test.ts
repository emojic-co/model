import { expect, test } from "bun:test"

import {
  collapse,
  emojiLeaderboard,
  pickPalette,
  shuffleSeeded,
} from "./regen.ts"

const P = (a: string, b: string, f: string) => ({ bg: [a, b], fg: f })

test("collapse unions emojis and styles across rows with the same normalized text", () => {
  const out = collapse([
    { text: "Bus is late", emojis: "🚌", styles: ["Irritated"], ...P("#111111", "#222222", "#eeeeee") },
    { text: "  bus   is late  ", emojis: "😤 🚌", styles: ["Tense", "Irritated"], ...P("#333333", "#444444", "#dddddd") },
  ])
  expect(out).toHaveLength(1)
  expect(out[0].emojis.split(" ").sort()).toEqual(["😤", "🚌"].sort())
  expect(out[0].styles.sort()).toEqual(["Irritated", "Tense"])
  expect(out[0].bg).toHaveLength(2)
  expect(typeof out[0].fg).toBe("string")
})

test("collapse drops rows that normalize to empty", () => {
  const out = collapse([
    { text: "😀😀😀", emojis: "😀", styles: ["Joyful"], ...P("#111111", "#222222", "#eeeeee") },
    { text: "real text here", emojis: "", styles: ["Deadpan"], ...P("#111111", "#222222", "#eeeeee") },
  ])
  expect(out.map((r) => r.text)).toEqual(["real text here"])
})

test("collapse keeps only styles in the closed set", () => {
  const out = collapse([
    { text: "hello world", emojis: "", styles: ["Joyful", "Bogus", "notreal"], ...P("#111111", "#222222", "#eeeeee") },
  ])
  expect(out[0].styles).toEqual(["Joyful"])
})

test("collapse emits a record with no bg/fg when no source row had a palette", () => {
  const out = collapse([{ text: "no colors", emojis: "🎈", styles: ["Playful"] }])
  expect(out).toHaveLength(1)
  expect("bg" in out[0]).toBe(false)
  expect("fg" in out[0]).toBe(false)
})

test("collapse keeps the first-seen raw text for a merged key", () => {
  const out = collapse([
    { text: "Hello  World", emojis: "😀", styles: [] },
    { text: "hello world", emojis: "🌍", styles: [] },
  ])
  expect(out).toHaveLength(1)
  expect(out[0].text).toBe("Hello  World")
  expect(out[0].emojis.split(" ").sort()).toEqual(["😀", "🌍"].sort())
})

test("pickPalette is deterministic and returns one of the given palettes", () => {
  const palettes = [
    P("#aaaaaa", "#bbbbbb", "#000000"),
    P("#cccccc", "#dddddd", "#111111"),
    P("#eeeeee", "#ffffff", "#222222"),
  ]
  const a = pickPalette("some key", palettes)
  expect(pickPalette("some key", palettes)).toEqual(a!)
  expect(palettes).toContainEqual(a!)
})

test("pickPalette returns undefined when there are no palettes", () => {
  expect(pickPalette("k", [])).toBeUndefined()
})

test("shuffleSeeded is a deterministic permutation and does not mutate input", () => {
  const xs = Array.from({ length: 60 }, (_, i) => i)
  const a = shuffleSeeded(xs, 42)
  expect(shuffleSeeded(xs, 42)).toEqual(a)
  expect([...a].sort((p, q) => p - q)).toEqual(xs)
  expect(a).not.toEqual(xs)
  expect(shuffleSeeded(xs, 7)).not.toEqual(a)
  expect(xs).toEqual(Array.from({ length: 60 }, (_, i) => i))
})

test("emojiLeaderboard ranks by row frequency, ties keep first-seen order", () => {
  const recs = [
    { emojis: "🍕 🍕 🚗" },
    { emojis: "🍕 🎂" },
    { emojis: "🚗" },
    { emojis: "🎸" },
  ]
  expect(emojiLeaderboard(recs, 3)).toEqual(["🍕", "🚗", "🎂"])
})
