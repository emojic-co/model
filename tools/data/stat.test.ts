import { expect, test } from "bun:test"

import { computeStats } from "./stat.ts"

const R = (text: string, emojis: string, styles: string[] = [], extra: Record<string, unknown> = {}) => ({
  text,
  emojis,
  styles,
  ...extra,
})

test("computeStats collapses raw rows by normalized text and counts records per text", () => {
  const s = computeStats([
    R("bus is late", "🚌", ["Irritated"]),
    R("  bus   is   late ", "😤 🚌", ["Tense"]),
    R("something else", "🐈", ["Joyful"]),
  ])
  expect(s.rawRows).toBe(3)
  expect(s.texts).toBe(2)
  const busRow = s.disagreement.find((d) => d.text === "bus is late")
  expect(busRow?.records).toBe(2)
  expect(busRow?.emojis).toEqual(["😤", "🚌"])
  expect(busRow?.uniqueEmojis).toBe(2)
})

test("computeStats buckets texts by unique emoji count with pct and ascending cumulative pct", () => {
  const s = computeStats([
    R("no emoji here", ""),
    R("one emoji a", "🐈"),
    R("one emoji b", "🐕"),
    R("two emojis c", "🌧 ☔"),
  ])
  expect(s.emojiPerText).toEqual([
    { count: 0, texts: 1, pct: 25, cumPct: 0, tailPct: 100 },
    { count: 1, texts: 2, pct: 50, cumPct: 25, tailPct: 75 },
    { count: 2, texts: 1, pct: 25, cumPct: 75, tailPct: 25 },
  ])
})

test("computeStats reports zero-emoji texts and means over distinct texts", () => {
  const s = computeStats([
    R("alpha line", ""),
    R("beta line", "🐈 🐕"),
    R("gamma line", "🌊"),
  ])
  expect(s.zeroEmojiTexts).toBe(1)
  expect(s.meanEmojisPerText).toBeCloseTo(1)
  expect(s.distinctEmojis).toBe(3)
})

test("computeStats sorts topEmojis by number of texts descending", () => {
  const s = computeStats([
    R("t one", "🔥"),
    R("t two", "🔥 ✨"),
    R("t three", "🔥 ✨ 🎉"),
  ])
  expect(s.topEmojis.map((e) => [e.emoji, e.texts])).toEqual([
    ["🔥", 3],
    ["✨", 2],
    ["🎉", 1],
  ])
})

test("computeStats counts the full closed style set and ignores out-of-set styles", () => {
  const s = computeStats([
    R("row a", "", ["Joyful", "Bogus"]),
    R("row b", "", ["Joyful"]),
    R("row c", "", ["Deadpan"]),
  ])
  expect(s.styleCounts).toHaveLength(21)
  const joyful = s.styleCounts.find((x) => x.style === "Joyful")
  expect(joyful?.texts).toBe(2)
  expect(s.styleCounts.some((x) => x.style === "Bogus")).toBe(false)
  expect(s.styleCounts[0].style).toBe("Joyful")
})

test("computeStats reports min/median/p90/max of normalized text length", () => {
  const s = computeStats([
    R("ab", ""),
    R("abcd", ""),
    R("abcdef", ""),
    R("abcdefgh", ""),
    R("abcdefghij", ""),
  ])
  expect(s.textLen).toEqual({ min: 2, median: 6, p90: 10, max: 10 })
})

test("computeStats lists highest-disagreement texts first", () => {
  const s = computeStats([
    R("calm text", "🙂"),
    R("messy text", "🐈"),
    R("messy text", "🐕 🦊"),
    R("messy text", "🐢"),
  ])
  expect(s.disagreement[0].text).toBe("messy text")
  expect(s.disagreement[0].uniqueEmojis).toBe(4)
})

test("computeStats detects language from script when untagged, and counts explicit row.lang as tagged", () => {
  const s = computeStats([
    R("hello there", "🙂"),
    R("שלום עולם", "🙂"),
    R("bonjour le monde", "🙂", [], { lang: "he" }),
  ])
  const en = s.langCounts.find((x) => x.lang === "en")
  const he = s.langCounts.find((x) => x.lang === "he")
  expect(en).toEqual({ lang: "en", texts: 1, pct: expect.closeTo(33.33, 1), tagged: 0 })
  expect(he).toEqual({ lang: "he", texts: 2, pct: expect.closeTo(66.67, 1), tagged: 1 })
})

test("computeStats averages distinct color palettes per text, dedupes repeats across records", () => {
  const s = computeStats([
    { text: "no colors", emojis: "", styles: [] },
    { text: "one color", emojis: "", styles: [], colors: [{ bg: ["#111111", "#222222"], fg: "#ffffff" }] },
    { text: "two colors", emojis: "", styles: [], colors: [
      { bg: ["#111111", "#222222"], fg: "#ffffff" },
      { bg: ["#333333", "#444444"], fg: "#000000" },
    ] },
    { text: "two colors", emojis: "", styles: [], colors: [{ bg: ["#111111", "#222222"], fg: "#ffffff" }] },
  ])
  expect(s.meanColorsPerText).toBeCloseTo(1)
})

test("computeStats estimates a color regressor R^2 ceiling from within-text vs. dataset color variance", () => {
  const s = computeStats([
    { text: "steady", emojis: "", styles: [], colors: [{ bg: ["#000000", "#000000"], fg: "#ffffff" }] },
    { text: "steady", emojis: "", styles: [], colors: [{ bg: ["#000000", "#000000"], fg: "#ffffff" }] },
    { text: "flip a", emojis: "", styles: [], colors: [
      { bg: ["#000000", "#000000"], fg: "#000000" },
      { bg: ["#ffffff", "#ffffff"], fg: "#ffffff" },
    ] },
    { text: "flip b", emojis: "", styles: [], colors: [
      { bg: ["#000000", "#000000"], fg: "#000000" },
      { bg: ["#ffffff", "#ffffff"], fg: "#ffffff" },
    ] },
  ])
  expect(s.colorFit.colorTexts).toBe(3)
  expect(s.colorFit.multiColorTexts).toBe(2)
  expect(s.colorFit.withinTextVariance).toBeGreaterThan(0)
  expect(s.colorFit.datasetVariance).toBeGreaterThan(0)
  expect(s.colorFit.r2Ceiling).toBeGreaterThanOrEqual(0)
  expect(s.colorFit.r2Ceiling).toBeLessThanOrEqual(1)
  expect(s.colorFit.r2Ceiling).toBeCloseTo(0, 5)
})

test("computeStats gives a color regressor R^2 ceiling of 1 when no text has color disagreement", () => {
  const s = computeStats([
    { text: "a", emojis: "", styles: [], colors: [{ bg: ["#111111", "#222222"], fg: "#ffffff" }] },
    { text: "b", emojis: "", styles: [], colors: [{ bg: ["#333333", "#444444"], fg: "#000000" }] },
  ])
  expect(s.colorFit.multiColorTexts).toBe(0)
  expect(s.colorFit.withinTextVariance).toBe(0)
  expect(s.colorFit.r2Ceiling).toBe(1)
})

test("computeStats counts neg, meta, and meta.single-emoji extra fields", () => {
  const s = computeStats([
    R("plain row", "🐈"),
    R("neg row", "🐈", [], { neg: "true" }),
    R("meta row", "🐈", [], { meta: { date: "2026-01-01" } }),
    R("single row", "🐈", [], { meta: { date: "2026-01-01", "single-emoji": true } }),
  ])
  expect(s.extraFields).toEqual({ neg: 1, meta: 2, singleEmoji: 1 })
})
