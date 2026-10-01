import { describe, expect, test } from 'vitest'
import { FEELINGS } from './feelings'
import { I18N, LANGS, translate, feelingLabel, readCookieLang, detectLang, langCookie } from './i18n'

const placeholders = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()

describe('strings file', () => {
  test('en and he have identical keys and placeholders', () => {
    expect(Object.keys(I18N.he).sort()).toEqual(Object.keys(I18N.en).sort())
    for (const k of Object.keys(I18N.en)) {
      expect(placeholders(I18N.he[k]), k).toEqual(placeholders(I18N.en[k]))
    }
  })
  test('every feeling has a label in both languages', () => {
    for (const f of Object.keys(FEELINGS)) for (const l of LANGS) expect(I18N[l][`feeling.${f}`], `${l}:${f}`).toBeTruthy()
  })
  test('literals jpg/gif/mp4 survive translation', () => {
    expect(I18N.he['card.copyJpg']).toContain('jpg')
    expect(I18N.he['card.saveGif']).toContain('gif')
    expect(I18N.he['card.saveMp4']).toContain('mp4')
    expect(I18N.he['settings.exportRes']).toContain('jpg / mp4')
  })
})

describe('translate', () => {
  test('substitutes placeholders', () => {
    expect(translate('en', 'settings.maxEmojis', { n: 7 })).toBe('Max emojis shown: 7')
  })
  test('missing placeholder var is left intact', () => {
    expect(translate('en', 'color.aria', {})).toBe('color {n}')
  })
  test('values containing braces are not re-substituted', () => {
    expect(translate('en', 'toast.saved', { format: '{n}', n: 'x' })).toBe('{n} saved ✓')
  })
  test('falls back to en, then to the key', () => {
    expect(translate('he', 'no.such.key')).toBe('no.such.key')
    expect(translate('fr', 'toast.cancelled')).toBe('cancelled')
  })
  test('feelingLabel falls back to the identifier for unknown feelings', () => {
    expect(feelingLabel('he', 'Joyful')).toBe('שמח')
    expect(feelingLabel('he', 'Brand New')).toBe('Brand New')
  })
})

describe('language detection', () => {
  test('reads the lang cookie among others', () => {
    expect(readCookieLang('a=1; lang=he; b=2')).toBe('he')
  })
  test('rejects unsupported / garbage / empty cookie values', () => {
    expect(readCookieLang('lang=fr')).toBeNull()
    expect(readCookieLang('lang=')).toBeNull()
    expect(readCookieLang(';;')).toBeNull()
    expect(readCookieLang(undefined)).toBeNull()
  })
  test('cookie wins over navigator; navigator he-IL / iw give he; else en', () => {
    expect(detectLang('lang=en', 'he-IL')).toBe('en')
    expect(detectLang('', 'he-IL')).toBe('he')
    expect(detectLang('', 'iw')).toBe('he')
    expect(detectLang('lang=fr', 'fr-FR')).toBe('en')
    expect(detectLang(undefined, undefined)).toBe('en')
  })
  test('cookie string is a year-long Lax cookie', () => {
    expect(langCookie('he')).toBe('lang=he; path=/; max-age=31536000; SameSite=Lax')
  })
})
