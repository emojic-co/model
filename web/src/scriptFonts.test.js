import { describe, it, expect } from 'vitest'
import { CLUSTERS } from './feelings'
import { scriptForLang, fontForScript, ensureScriptFontsLoaded, LATIN } from './scriptFonts'

const SCRIPTS = [
  'cyrillic', 'greek', 'arabic', 'hebrew', 'devanagari', 'thai', 'japanese', 'korean', 'chinese',
]

describe('scriptForLang', () => {
  it('maps known non-latin languages to their script', () => {
    expect(scriptForLang('ja')).toBe('japanese')
    expect(scriptForLang('ko')).toBe('korean')
    expect(scriptForLang('zh')).toBe('chinese')
    expect(scriptForLang('zh-Hant')).toBe('chinese')
    expect(scriptForLang('ar')).toBe('arabic')
    expect(scriptForLang('he')).toBe('hebrew')
    expect(scriptForLang('ru')).toBe('cyrillic')
    expect(scriptForLang('bg')).toBe('cyrillic')
    expect(scriptForLang('uk')).toBe('cyrillic')
    expect(scriptForLang('el')).toBe('greek')
    expect(scriptForLang('hi')).toBe('devanagari')
    expect(scriptForLang('mr')).toBe('devanagari')
    expect(scriptForLang('th')).toBe('thai')
  })

  it('defaults everything else to latin, including uncurated non-latin scripts', () => {
    expect(scriptForLang('en')).toBe(LATIN)
    expect(scriptForLang('fr')).toBe(LATIN)
    expect(scriptForLang('bn')).toBe(LATIN)
    expect(scriptForLang(undefined)).toBe(LATIN)
  })
})

describe('fontForScript', () => {
  it('returns a distinct quoted font family for every curated (script, cluster) pair', () => {
    for (const script of SCRIPTS) {
      for (const cluster of Object.keys(CLUSTERS)) {
        expect(fontForScript(script, cluster), `${script}/${cluster}`).toMatch(/^".+", .+/)
      }
    }
  })

  it('falls back to a generic multilingual font for an unknown script or cluster', () => {
    expect(fontForScript('bengali', 'joy')).toMatch(/Noto Sans/)
    expect(fontForScript('japanese', 'nope')).toMatch(/Noto Sans/)
  })
})

describe('ensureScriptFontsLoaded', () => {
  it('never throws, even without a document (node test env) or for the latin bucket', () => {
    expect(() => ensureScriptFontsLoaded(LATIN)).not.toThrow()
    expect(() => ensureScriptFontsLoaded('japanese')).not.toThrow()
    expect(() => ensureScriptFontsLoaded('not-a-script')).not.toThrow()
  })
})
