import { expect, test } from "bun:test"

import {
  collapse,
  emojiVocab,
  greedyCap,
  pickPalette,
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

const R = (emojis: string, text = emojis): { text: string; emojis: string; styles: string[] } => ({
  text,
  emojis,
  styles: [],
})

test("greedyCap keeps rows until an emoji hits max-count, then drops only rows carrying it", () => {
  const records = [
    ...Array.from({ length: 8 }, (_, i) => R("🍕", `pizza ${i}`)),
    ...Array.from({ length: 3 }, (_, i) => R("🚗", `car ${i}`)),
  ]
  const { kept, counts, dropped } = greedyCap(records, 5)
  expect(kept.filter((r) => r.emojis === "🍕")).toHaveLength(5)
  expect(kept.filter((r) => r.emojis === "🚗")).toHaveLength(3)
  expect(counts.get("🍕")).toBe(5)
  expect(counts.get("🚗")).toBe(3)
  expect(dropped).toBe(3)
})

test("greedyCap drops a multi-emoji row once any of its emojis is at max-count, but keeps its other emoji's count from a later solo row", () => {
  const records = [
    ...Array.from({ length: 5 }, (_, i) => R("🍕", `pizza ${i}`)),
    R("🍕 🚗", "combo"),
    R("🚗", "car alone"),
  ]
  const { kept, counts } = greedyCap(records, 5)
  expect(kept.map((r) => r.text)).toEqual([
    "pizza 0", "pizza 1", "pizza 2", "pizza 3", "pizza 4", "car alone",
  ])
  expect(counts.get("🍕")).toBe(5)
  expect(counts.get("🚗")).toBe(1)
})

test("greedyCap keeps emoji-less rows unconditionally and counts each row's emoji once even if repeated", () => {
  const records = [R("🍕 🍕", "double pizza"), R("", "no emoji")]
  const { kept, counts } = greedyCap(records, 1)
  expect(kept.map((r) => r.text)).toEqual(["double pizza", "no emoji"])
  expect(counts.get("🍕")).toBe(1)
})

test("greedyCap does not mutate its input", () => {
  const records = Array.from({ length: 3 }, (_, i) => R("🍕", `pizza ${i}`))
  greedyCap(records, 1)
  expect(records).toHaveLength(3)
})

test("emojiVocab keeps only emojis at or above min-count, ranked by count descending, ties first-seen", () => {
  const counts = new Map([
    ["🍕", 6],
    ["🚗", 3],
    ["🎸", 6],
    ["🎂", 2],
  ])
  expect(emojiVocab(counts, 3)).toEqual(["🍕", "🎸", "🚗"])
})

test("emojiVocab returns an empty vocab when nothing reaches min-count", () => {
  const counts = new Map([["🍕", 2]])
  expect(emojiVocab(counts, 3)).toEqual([])
})
