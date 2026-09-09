import { expect, test } from "bun:test"

import { stripVS } from "../analysis/cldr-baseline.ts"
import { buildFlexRanker } from "./flexrank.ts"

test("buildJson emits vocab keywords + idf map consistent with the ranker", async () => {
  const { rank, buildJson } = await buildFlexRanker(32)
  const vocabOrig = ["🍕", "🐞", "🧁"]
  const vocab = new Map(vocabOrig.map((e) => [stripVS(e), e]))
  const j = buildJson(vocabOrig)

  expect(j.emojis).toEqual(vocabOrig)
  expect(j.keywords.length).toBe(3)
  expect(j.keywords.every((ks) => Array.isArray(ks) && ks.length > 0)).toBe(true)
  expect(typeof j.idf_default).toBe("number")

  for (const ks of j.keywords) for (const k of ks) expect(k in j.idf).toBe(true)

  const before = rank("cheese pizza time", vocab)
  expect(before.flexsearch[0][0]).toBe("🍕")
})
