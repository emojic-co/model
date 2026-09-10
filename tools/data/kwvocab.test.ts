import { describe, expect, it } from "bun:test"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

import { keywordVocabFromIiJson, keywordVocabFromReport } from "./kwvocab.ts"

describe("keywordVocabFromIiJson", () => {
  const v = keywordVocabFromIiJson(["\u{1F600}", "\u{1F355}", "\u{1F389}"])
  it("keys are 3-6 char single lowercase alnum tokens", () => {
    for (const k of v) {
      expect(k).toMatch(/^[a-z0-9]+$/)
      expect(k.length).toBeGreaterThanOrEqual(3)
      expect(k.length).toBeLessThanOrEqual(6)
    }
  })
  it("deduped, alphabetical, capped at 1000", () => {
    expect(new Set(v).size).toBe(v.length)
    expect([...v].sort()).toEqual(v)
    expect(v.length).toBeLessThanOrEqual(1000)
  })
  it("empty emoji set -> empty vocab", () => {
    expect(keywordVocabFromIiJson([])).toEqual([])
  })
})

describe("keywordVocabFromReport", () => {
  it("null when the dir has no report.json", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      expect(keywordVocabFromReport(d)).toBeNull()
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
  it("null when dir does not exist", () => {
    expect(keywordVocabFromReport(join(tmpdir(), "nope-does-not-exist"))).toBeNull()
  })
  it("newest folder's ranked, top-1000, alphabetical", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      mkdirSync(join(d, "26-01-01-00-00-aaa"))
      mkdirSync(join(d, "26-02-02-00-00-bbb"))
      writeFileSync(
        join(d, "26-01-01-00-00-aaa", "report.json"),
        JSON.stringify({ keywords_flex: { ranked: [{ kw: "old", rank: 99 }] } }),
      )
      writeFileSync(
        join(d, "26-02-02-00-00-bbb", "report.json"),
        JSON.stringify({
          keywords_flex: {
            ranked: [
              { kw: "zebra", rank: 50 },
              { kw: "apple", rank: 40 },
            ],
          },
        }),
      )
      expect(keywordVocabFromReport(d)).toEqual(["apple", "zebra"])
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
  it("null when the newest report lacks keywords_flex", () => {
    const d = mkdtempSync(join(tmpdir(), "rep-"))
    try {
      mkdirSync(join(d, "26-03-03-00-00-ccc"))
      writeFileSync(join(d, "26-03-03-00-00-ccc", "report.json"), JSON.stringify({ data: {} }))
      expect(keywordVocabFromReport(d)).toBeNull()
    } finally {
      rmSync(d, { recursive: true, force: true })
    }
  })
})
