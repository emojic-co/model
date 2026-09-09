import { expect, test } from "bun:test"

import { minimalLine, minimize } from "./minimize.ts"

const P = (a: string, b: string, f: string) => ({ bg: [a, b], fg: f })

test("minimize collapses by normalized text, unions emojis/styles, last palette wins", () => {
  const lines = minimize(
    [
      { text: "Bus is late", emojis: "🚌", styles: ["Irritated"], ...P("#111111", "#222222", "#eeeeee") },
      { text: "  bus   is late  ", emojis: "😤 🚌", styles: ["Tense", "Irritated"], ...P("#333333", "#444444", "#dddddd") },
    ],
    "abc1234",
  )
    .trimEnd()
    .split("\n")
  expect(lines).toHaveLength(1)
  const rec = JSON.parse(lines[0])
  expect(rec.emojis.split(" ").sort()).toEqual(["😤", "🚌"].sort())
  expect(rec.styles.sort()).toEqual(["Irritated", "Tense"])
  expect(rec.bg).toEqual(["#333333", "#444444"])
  expect(rec.fg).toBe("#dddddd")
  expect(rec.min).toBe("abc1234")
})

test("minimize strips every non-base extra field, keeping only text/emojis/styles/bg/fg/min", () => {
  const out = minimize(
    [
      {
        text: "sky at dusk",
        emojis: "🌇",
        styles: ["Wistful"],
        ...P("#111111", "#222222", "#eeeeee"),
        meta: { src: "colors", date: "2026-01-01" },
        color: "orange",
        neg: "true",
        remojis: ["x"],
        keyword: "dusk",
      },
    ],
    "deadbee",
  )
  const rec = JSON.parse(out.trim())
  expect(Object.keys(rec).sort()).toEqual(["bg", "emojis", "fg", "min", "styles", "text"])
  expect(rec.min).toBe("deadbee")
})

test("minimize omits bg/fg when no source row carried a palette", () => {
  const out = minimize([{ text: "no colors", emojis: "🎈", styles: ["Playful"] }], "abc1234")
  const rec = JSON.parse(out.trim())
  expect(Object.keys(rec).sort()).toEqual(["emojis", "min", "styles", "text"])
})

test("minimalLine marks one collapsed record with the short sha and drops its extra", () => {
  const line = minimalLine(
    {
      text: "roses at dawn",
      emojis: "🌹",
      styles: ["Wistful"],
      bg: ["#111111", "#222222"],
      fg: "#eeeeee",
      extra: { color: "red" },
    },
    "1234abc",
  )
  expect(JSON.parse(line)).toEqual({
    text: "roses at dawn",
    emojis: "🌹",
    styles: ["Wistful"],
    bg: ["#111111", "#222222"],
    fg: "#eeeeee",
    min: "1234abc",
  })
})

test("minimize drops rows that normalize to empty and preserves first-seen order", () => {
  const texts = minimize(
    [
      { text: "😀😀😀", emojis: "😀", styles: ["Joyful"] },
      { text: "zebra", emojis: "🦓", styles: [] },
      { text: "apple", emojis: "🍎", styles: [] },
    ],
    "sha",
  )
    .trimEnd()
    .split("\n")
    .map((l) => JSON.parse(l).text)
  expect(texts).toEqual(["zebra", "apple"])
})

test("minimize refreshes an existing min marker to the current sha", () => {
  const out = minimize(
    [{ text: "old row", emojis: "📦", styles: ["Deadpan"], ...P("#111111", "#222222", "#eeeeee"), min: "oldsha0" }],
    "newsha1",
  )
  expect(JSON.parse(out.trim()).min).toBe("newsha1")
})
