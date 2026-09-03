import { expect, test } from "bun:test"

import {
  collapse,
  downsampleEmojis,
  emojiLeaderboard,
  pickPalette,
  shuffleSeeded,
  toLine,
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

test("collapse carries non-base fields through as extra, first-seen wins", () => {
  const out = collapse([
    { text: "sky at dusk", emojis: "🌇", styles: ["Wistful"], ...P("#111111", "#222222", "#eeeeee"), color: "orange" },
    { text: "  sky at dusk ", emojis: "", styles: ["Deadpan"], color: "red", note: "second" },
  ])
  expect(out).toHaveLength(1)
  expect(out[0].extra).toEqual({ color: "orange", note: "second" })
})

test("collapse leaves extra unset when rows carry only base fields", () => {
  const out = collapse([
    { text: "plain row", emojis: "", styles: ["Deadpan"], ...P("#111111", "#222222", "#eeeeee") },
  ])
  expect("extra" in out[0]).toBe(false)
})

test("toLine emits extra fields after the base schema fields", () => {
  const line = toLine({
    text: "roses at dawn",
    emojis: "🌹",
    styles: ["Wistful"],
    bg: ["#111111", "#222222"],
    fg: "#eeeeee",
    extra: { color: "red" },
  })
  expect(line).toBe(
    '{"text":"roses at dawn","emojis":"🌹","styles":["Wistful"],'
    + '"bg":["#111111","#222222"],"fg":"#eeeeee","color":"red"}',
  )
  expect(JSON.parse(toLine({ text: "t", emojis: "", styles: [] }))).toEqual({
    text: "t",
    emojis: "",
    styles: [],
  })
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

const R = (emojis: string, text = emojis): { text: string; emojis: string; styles: string[] } => ({
  text,
  emojis,
  styles: [],
})

test("downsampleEmojis caps a top emoji at ratio x the least frequent leaderboard emoji", () => {
  const records = [
    ...Array.from({ length: 20 }, (_, i) => R("🍕", `pizza ${i}`)),
    ...Array.from({ length: 3 }, (_, i) => R("🚗", `car ${i}`)),
    R("🎸", "guitar"),
  ]
  const { kept, cap, minFreq, minEmoji, dropped } = downsampleEmojis(
    records,
    ["🍕", "🚗", "🎸"],
    5,
    42,
  )
  expect({ cap, minFreq, minEmoji }).toEqual({ cap: 5, minFreq: 1, minEmoji: "🎸" })
  expect(kept.filter((r) => r.emojis === "🍕")).toHaveLength(5)
  expect(kept.filter((r) => r.emojis === "🚗")).toHaveLength(3)
  expect(kept.filter((r) => r.emojis === "🎸")).toHaveLength(1)
  expect(dropped).toBe(records.length - kept.length)
  expect(dropped).toBe(15)
})

test("downsampleEmojis drops the whole multi-label row when one of its emojis is over cap", () => {
  const records = [
    ...Array.from({ length: 30 }, (_, i) => R("🍕", `pizza ${i}`)),
    ...Array.from({ length: 2 }, (_, i) => R(`🍕 🚗`, `combo ${i}`)),
  ]
  const { kept, cap } = downsampleEmojis(records, ["🍕", "🚗"], 5, 42)
  const count = (e: string) =>
    kept.filter((r) => r.emojis.split(" ").includes(e)).length
  expect(cap).toBe(10)
  expect(count("🍕")).toBeLessThanOrEqual(10)
  expect(count("🚗")).toBeLessThanOrEqual(10)
  for (const r of kept) {
    expect(r.emojis.split(" ").filter((e) => e === "🍕" || e === "🚗").length).toBeGreaterThan(0)
  }
})

test("downsampleEmojis is deterministic for a fixed seed and leaves the input unmutated", () => {
  const records = [
    ...Array.from({ length: 12 }, (_, i) => R("🍕", `pizza ${i}`)),
    ...Array.from({ length: 2 }, (_, i) => R("🎂", `cake ${i}`)),
  ]
  const a = downsampleEmojis(records, ["🍕", "🎂"], 3, 7).kept.map((r) => r.text)
  const b = downsampleEmojis(records, ["🍕", "🎂"], 3, 7).kept.map((r) => r.text)
  expect(b).toEqual(a)
  expect(records).toHaveLength(14)
})

test("downsampleEmojis is a no-op when a leaderboard emoji never occurs (cap 0)", () => {
  const records = Array.from({ length: 8 }, (_, i) => R("🍕", `pizza ${i}`))
  const { kept, cap, dropped } = downsampleEmojis(records, ["🍕", "🌵"], 5, 42)
  expect({ cap, dropped }).toEqual({ cap: 0, dropped: 0 })
  expect(kept).toHaveLength(8)
})
