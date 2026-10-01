import { describe, it, expect } from 'vitest'
import { CLUSTERS } from './feelings'
import { scriptForLang, fontForScript, ensureScriptFontsLoaded, LATIN } from './scriptFonts'

const SCRIPTS = [
  'hebrew',
]

describe('scriptForLang', () => {
  it('maps Hebrew to its script', () => {
    expect(scriptForLang('he')).toBe('hebrew')
  })

  it('defaults everything else to latin, including uncurated non-latin scripts', () => {
    expect(scriptForLang('en')).toBe(LATIN)
    expect(scriptForLang('fr')).toBe(LATIN)
    expect(scriptForLang('ja')).toBe(LATIN)
    expect(scriptForLang(undefined)).toBe(LATIN)
  })
})

describe('fontForScript', () => {
  it('returns a quoted font family for every curated (script, cluster) pair', () => {
    for (const script of SCRIPTS) {
      for (const cluster of Object.keys(CLUSTERS)) {
        expect(fontForScript(script, cluster), `${script}/${cluster}`).toMatch(/^".+", .+/)
      }
    }
  })

  it('falls back to a generic multilingual font for an unknown script or cluster', () => {
    expect(fontForScript('japanese', 'joy')).toMatch(/Noto Sans/)
    expect(fontForScript('hebrew', 'nope')).toMatch(/Noto Sans/)
  })
})

describe('ensureScriptFontsLoaded', () => {
  it('never throws, even without a document (node test env) or for the latin bucket', () => {
    expect(() => ensureScriptFontsLoaded(LATIN)).not.toThrow()
    expect(() => ensureScriptFontsLoaded('hebrew')).not.toThrow()
    expect(() => ensureScriptFontsLoaded('not-a-script')).not.toThrow()
  })
})
