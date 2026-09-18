import { expect, test } from "bun:test"

import {
  Batcher,
  colorBatchPlan,
  countEmojis,
  countLangs,
  groupDeficits,
  langLine,
  langsSuffix,
  langSuffix,
  languageName,
  leastFrequentLang,
  lowestFreqEmojis,
  parseLang,
  rankMissingByFreq,
  resolveLangs,
  TOPICS,
  topicForBatch,
} from "./upsample.ts"

test("TOPICS is a non-trivial unique list", () => {
  expect(TOPICS.length).toBeGreaterThan(15)
  expect(new Set(TOPICS).size).toBe(TOPICS.length)
})

test("topicForBatch round-robins over TOPICS", () => {
  expect(topicForBatch(0)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length)).toBe(TOPICS[0])
  expect(topicForBatch(TOPICS.length + 1)).toBe(TOPICS[1])
})

test("countLangs trusts an explicit row.lang over script detection, and detects otherwise", () => {
  const counts = countLangs([
    { text: "hello there" },
    { text: "שלום עולם" },
    { text: "bonjour le monde", lang: "he" },
    { text: "another line", lang: "fr" },
  ])
  expect(counts.get("en")).toBe(2)
  expect(counts.get("he")).toBe(2)
})

test("leastFrequentLang picks the lowest-count language, breaking ties by LANGS order", () => {
  expect(leastFrequentLang(new Map([["en", 100], ["he", 5]]))).toBe("he")
  expect(leastFrequentLang(new Map())).toBe("en")
})

test("colorBatchPlan splits each colour's per-colour total into batches of at most batchSize", () => {
  const plan = colorBatchPlan(["red", "blue"], 100, 40)
  expect(plan).toEqual([
    { color: "red", n: 40 },
    { color: "red", n: 40 },
    { color: "red", n: 20 },
    { color: "blue", n: 40 },
    { color: "blue", n: 40 },
    { color: "blue", n: 20 },
  ])
})

test("colorBatchPlan emits one batch when per <= batchSize and sums each colour back to per", () => {
  const plan = colorBatchPlan(["red", "green", "blue"], 30, 50)
  expect(plan).toEqual([
    { color: "red", n: 30 },
    { color: "green", n: 30 },
    { color: "blue", n: 30 },
  ])
  const total = plan.filter((b) => b.color === "red").reduce((s, b) => s + b.n, 0)
  expect(total).toBe(30)
})

test("countEmojis counts distinct emojis per row over all of data, not just a vocab", () => {
  const rows = [
    { emojis: "📺 📺 🍕" },
    { emojis: "🍕 🚗" },
    { emojis: "🛰️" },
    {},
  ]
  const counts = countEmojis(rows)
  expect(counts.get("📺")).toBe(1)
  expect(counts.get("🍕")).toBe(2)
  expect(counts.get("🚗")).toBe(1)
  expect(counts.get("🛰️")).toBe(1)
  expect(counts.has("🎂")).toBe(false)
})

test("groupDeficits skips groups without a goals.yml target or already at/above it", () => {
  const groups = { a: ["1", "2", "3", "4"], b: ["5", "6"] }
  const vocab = ["1", "2", "5", "6"]
  const targets = { a: 0.5, c: 1 }
  expect(groupDeficits(groups, vocab, targets)).toEqual([])
})

test("groupDeficits reports the missing count needed to reach the goal fraction", () => {
  const groups = { a: ["1", "2", "3", "4"] }
  const vocab = ["1"]
  const targets = { a: 0.75 }
  // covered=1/4=0.25, need >=0.75 -> ceil(0.75*4 - 1) = 2 more
  expect(groupDeficits(groups, vocab, targets)).toEqual([
    { name: "a", needed: 2, missing: ["2", "3", "4"] },
  ])
})

test("groupDeficits caps needed at the number of missing candidates", () => {
  const groups = { a: ["1", "2"] }
  const vocab: string[] = []
  const targets = { a: 1 }
  expect(groupDeficits(groups, vocab, targets)).toEqual([
    { name: "a", needed: 2, missing: ["1", "2"] },
  ])
})

test("groupDeficits matches vocab and members ignoring variation selectors", () => {
  const groups = { a: ["😐️", "😑"] }
  const vocab = ["😐"]
  const targets = { a: 1 }
  expect(groupDeficits(groups, vocab, targets)).toEqual([
    { name: "a", needed: 1, missing: ["😑"] },
  ])
})

test("rankMissingByFreq ranks by frequency desc, ties broken deterministically by seed", () => {
  const counts = new Map([
    ["🍎", 5],
    ["🍌", 20],
  ])
  expect(rankMissingByFreq(["🍎", "🍌", "🍇"], counts)).toEqual(["🍌", "🍎", "🍇"])
})

test("rankMissingByFreq treats emoji absent from counts as freq 0", () => {
  const counts = new Map([["🍎", 5]])
  const ranked = rankMissingByFreq(["🍇", "🍎", "🥝"], counts)
  expect(ranked[0]).toBe("🍎")
  expect(new Set(ranked.slice(1))).toEqual(new Set(["🍇", "🥝"]))
})

test("rankMissingByFreq is stable for a fixed seed", () => {
  const counts = new Map<string, number>()
  const a = rankMissingByFreq(["🍇", "🥝", "🍉"], counts, 1)
  const b = rankMissingByFreq(["🍇", "🥝", "🍉"], counts, 1)
  expect(a).toEqual(b)
})

test("lowestFreqEmojis returns the lowest fraction by count, rounded and ties by first-seen", () => {
  const counts = new Map([
    ["a", 5],
    ["b", 0],
    ["c", 2],
    ["d", 0],
    ["e", 10],
    ["f", 3],
    ["g", 1],
    ["h", 4],
    ["i", 8],
    ["j", 9],
  ])
  // 10 emoji * 0.1 = 1 -> lowest one, ties broken by first-seen
  expect(lowestFreqEmojis(counts, 0.1)).toEqual(["b"])
  // 10 emoji * 0.3 = 3 -> lowest three
  expect(lowestFreqEmojis(counts, 0.3)).toEqual(["b", "d", "g"])
})

test("lowestFreqEmojis always selects at least one emoji", () => {
  const counts = new Map([["a", 5], ["b", 1]])
  expect(lowestFreqEmojis(counts, 0.1)).toEqual(["b"])
})

test("languageName resolves a known ISO code and falls back to the code otherwise", () => {
  expect(languageName("he")).toBe("Hebrew")
  expect(languageName("not-a-real-code")).toBe("not-a-real-code")
})

test("langLine is empty for no language and names the language otherwise", () => {
  expect(langLine(undefined)).toEqual([])
  expect(langLine("he").join(" ")).toContain("Hebrew")
})

test("langSuffix is empty for no language and shows the code otherwise", () => {
  expect(langSuffix(undefined)).toBe("")
  expect(langSuffix("he")).toBe(" (lang: he)")
})

test("parseLang trims/lowercases a code and treats blank/undefined as English", () => {
  expect(parseLang(undefined)).toBeUndefined()
  expect(parseLang("")).toBeUndefined()
  expect(parseLang("  HE ")).toBe("he")
})

test("resolveLangs returns every known language when none is given, or just the one given", () => {
  expect(resolveLangs(undefined)).toEqual(["en", "he"])
  expect(resolveLangs("he")).toEqual(["he"])
})

test("langsSuffix shows every language when more than one, otherwise falls back to langSuffix", () => {
  expect(langsSuffix(["en", "he"])).toBe(" (langs: en, he)")
  expect(langsSuffix(["he"])).toBe(" (lang: he)")
})

test("Batcher emits a batch only once it reaches the configured size", () => {
  const b = new Batcher<number>(3)
  expect(b.push(1)).toBeNull()
  expect(b.push(2)).toBeNull()
  expect(b.push(3)).toEqual([1, 2, 3])
  expect(b.push(4)).toBeNull()
})

test("Batcher.flush returns the partial batch, or null when empty", () => {
  const b = new Batcher<number>(3)
  expect(b.flush()).toBeNull()
  b.push(1)
  expect(b.flush()).toEqual([1])
  expect(b.flush()).toBeNull()
})

test("Batcher starts a fresh buffer after each emitted batch", () => {
  const b = new Batcher<number>(2)
  expect(b.push(1)).toBeNull()
  expect(b.push(2)).toEqual([1, 2])
  expect(b.push(3)).toBeNull()
  expect(b.push(4)).toEqual([3, 4])
})
