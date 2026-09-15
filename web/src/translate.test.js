import { describe, it, expect } from 'vitest'
import { translationSupported, supportedLanguages, languageName, detectAndTranslate } from './translate'

describe('translationSupported', () => {
  it('is false in a plain node environment with no Translator/LanguageDetector globals', () => {
    expect(translationSupported()).toBe(false)
  })
})

describe('supportedLanguages', () => {
  it('excludes english and sorts by display name', () => {
    const langs = supportedLanguages()
    expect(langs.find((l) => l.code === 'en')).toBeUndefined()
    const names = langs.map((l) => l.name)
    expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b)))
  })

  it('resolves known codes to readable names', () => {
    const langs = supportedLanguages()
    expect(langs.find((l) => l.code === 'ja')?.name).toBe('Japanese')
    expect(langs.find((l) => l.code === 'ar')?.name).toBe('Arabic')
    expect(langs.find((l) => l.code === 'zh-Hant')?.name).toBeTruthy()
  })
})

describe('languageName', () => {
  it('returns a readable display name for a BCP-47 code', () => {
    expect(languageName('fr')).toBe('French')
  })
})

describe('detectAndTranslate', () => {
  it('falls back to the original text, tagged unsupported, when the browser has no translation APIs', async () => {
    const result = await detectAndTranslate('こんにちは')
    expect(result).toEqual({
      text: 'こんにちは',
      lang: 'en',
      detectedLang: null,
      translated: false,
      unsupported: true,
    })
  })
})
