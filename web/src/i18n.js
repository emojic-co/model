// UI localization (strings: i18n.yml, the ground truth shared with Android via style.yml).
import { createContext, createElement, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { parse } from 'yaml'
import raw from './i18n.yml?raw'
import { ensureScriptFontsLoaded, scriptForLang } from './scriptFonts'

export const I18N = parse(raw)
export const LANGS = ['en', 'he']
export const LANG_NAMES = { en: 'English', he: 'עברית' }
const COOKIE = 'lang'
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365

function format(str, vars) {
  return str.replace(/\{(\w+)\}/g, (m, k) => (vars && k in vars ? String(vars[k]) : m))
}

export function translate(lang, key, vars) {
  const s = I18N[lang]?.[key] ?? I18N.en[key] ?? key
  return format(s, vars)
}

export function feelingLabel(lang, id) {
  return I18N[lang]?.[`feeling.${id}`] ?? I18N.en[`feeling.${id}`] ?? id
}

export function readCookieLang(cookieStr) {
  const m = /(?:^|;\s*)lang=([^;]*)/.exec(cookieStr ?? '')
  return m && LANGS.includes(m[1]) ? m[1] : null
}

export function detectLang(cookieStr, navLang) {
  const saved = readCookieLang(cookieStr)
  if (saved) return saved
  return /^(he|iw)(-|$)/i.test(navLang ?? '') ? 'he' : 'en'
}

export function langCookie(lang) {
  return `${COOKIE}=${lang}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`
}

const LangContext = createContext({
  lang: 'en',
  setLang: () => {},
  t: (key, vars) => translate('en', key, vars),
  feeling: (id) => feelingLabel('en', id),
})

export const useI18n = () => useContext(LangContext)

function initialLang() {
  try {
    return detectLang(document.cookie, navigator.language)
  } catch {
    return 'en'
  }
}

export function LangProvider({ children }) {
  const [lang, setLangState] = useState(initialLang)
  const setLang = useCallback((l) => {
    if (!LANGS.includes(l)) return
    setLangState(l)
    try {
      document.cookie = langCookie(l)
    } catch {}
  }, [])

  useEffect(() => {
    const root = document.documentElement
    root.lang = lang
    root.dir = lang === 'he' ? 'rtl' : 'ltr'
    ensureScriptFontsLoaded(scriptForLang(lang))
  }, [lang])

  const value = useMemo(
    () => ({
      lang,
      setLang,
      t: (key, vars) => translate(lang, key, vars),
      feeling: (id) => feelingLabel(lang, id),
    }),
    [lang, setLang],
  )
  return createElement(LangContext.Provider, { value }, children)
}
