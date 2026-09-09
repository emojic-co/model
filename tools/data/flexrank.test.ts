import { describe, expect, it } from "bun:test"

import { DATA_JSONL } from "../../files.ts"
import { KW_MAX_LEN, KW_MIN_LEN } from "./config"
import { buildFlexRanker } from "./flexrank.ts"
import { readJsonl } from "./io.ts"

const corpus = (await readJsonl<{ text?: unknown }>(DATA_JSONL))
  .map((r) => (typeof r.text === "string" ? r.text : ""))
  .filter(Boolean)

describe("kw ranker", () => {
  it("vocab: single-token, length-bounded, deduped, alphabetical", async () => {
    const r = await buildFlexRanker(corpus)
    expect(r.kwVocab.length).toBeGreaterThan(500)
    for (const k of r.kwVocab) {
      expect(k).toMatch(/^[a-z0-9]+$/)
      expect(k.length).toBeGreaterThanOrEqual(KW_MIN_LEN)
      expect(k.length).toBeLessThanOrEqual(KW_MAX_LEN)
    }
    expect(new Set(r.kwVocab).size).toBe(r.kwVocab.length)
    expect([...r.kwVocab].sort()).toEqual(r.kwVocab)
  })

  it("tfVec: exact hit = 1.0, miss = 0, dense length = |vocab|", async () => {
    const r = await buildFlexRanker(corpus)
    const kw = r.kwVocab[0]
    const v = r.tfVec(kw)
    expect(v.length).toBe(r.kwVocab.length)
    expect(v[0]).toBe(1.0)
    expect(r.tfVec("zzzznotawordzzzz").every((x) => x === 0)).toBe(true)
  })

  it("corpus filter: sampled vocab words appear in the corpus", async () => {
    const r = await buildFlexRanker(corpus)
    for (const k of [...r.kwVocab.slice(0, 3), ...r.kwVocab.slice(-3)]) {
      const hit = corpus.some((t) =>
        t
          .toLowerCase()
          .replace(/[^a-z0-9\s]/g, " ")
          .split(/\s+/)
          .includes(k),
      )
      expect(hit).toBe(true)
    }
  })

  it("buildJson emits only kw_vocab", async () => {
    const r = await buildFlexRanker(corpus)
    expect(Object.keys(r.buildJson())).toEqual(["kw_vocab"])
    expect(r.buildJson().kw_vocab).toEqual(r.kwVocab)
  })
})
