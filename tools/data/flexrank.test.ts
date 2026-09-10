import { describe, expect, it } from "bun:test"

import { buildFlexRanker } from "./flexrank.ts"

const VOCAB = ["apple", "banana", "cherry", "grape", "lemon"]

describe("flex ranker", () => {
  it("kwVocab echoes the given vocab", () => {
    expect(buildFlexRanker(VOCAB).kwVocab).toEqual(VOCAB)
  })

  it("tfVec: exact hit = 1.0, miss = 0, dense length = |vocab|", () => {
    const r = buildFlexRanker(VOCAB)
    const v = r.tfVec("apple")
    expect(v.length).toBe(VOCAB.length)
    expect(v[0]).toBe(1.0)
    expect(r.tfVec("zzzznotawordzzzz").every((x) => x === 0)).toBe(true)
  })

  it("tfVec: fuzzy prefix match scores min/max length ratio", () => {
    expect(buildFlexRanker(["apple"]).tfVec("apples")[0]).toBeCloseTo(0.833, 3)
  })

  it("buildJson emits only kw_vocab", () => {
    const r = buildFlexRanker(VOCAB)
    expect(Object.keys(r.buildJson())).toEqual(["kw_vocab"])
    expect(r.buildJson().kw_vocab).toEqual(VOCAB)
  })
})
