import { parse } from 'yaml'
import raw from './fonts.yml?raw'
export const LATIN = 'latin'

export const LANG_SCRIPT = {
  he: 'hebrew',
}

export function scriptForLang(lang) {
  return LANG_SCRIPT[lang] ?? LATIN
}

const HEBREW_RE = /[֐-׿]/

export function langForText(text) {
  return HEBREW_RE.test(text) ? 'he' : 'en'
}

export const FONTS = parse(raw)

export const cssFont = ({ family, generic }) => `"${family}", ${generic}`

const FALLBACK_FONT = `${cssFont(FONTS.fallback)}, ui-sans-serif, system-ui`

export function latinFont(feeling) {
  const f = FONTS.latin[feeling]
  return f ? cssFont(f) : FALLBACK_FONT
}

export function fontForScript(script, cluster) {
  const f = FONTS.scripts[script]?.[cluster]
  return f ? cssFont(f) : FALLBACK_FONT
}

function googleSpec({ family, weight, italic }) {
  const name = family.replaceAll(' ', '+')
  if (italic) return `family=${name}:ital,wght@1,${weight ?? 400}`
  return weight ? `family=${name}:wght@${weight}` : `family=${name}`
}

export const SCRIPT_FONT_QUERY = Object.fromEntries(
  Object.entries(FONTS.scripts).map(([script, clusters]) => [
    script,
    [...new Set(Object.values(clusters).map(googleSpec))].join('&'),
  ]),
)

const loadedScripts = new Set()

export function ensureScriptFontsLoaded(script) {
  if (script === LATIN || loadedScripts.has(script) || typeof document === 'undefined') return
  const query = SCRIPT_FONT_QUERY[script]
  if (!query) return
  loadedScripts.add(script)
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = `https://fonts.googleapis.com/css2?${query}&display=swap`
  document.head.appendChild(link)
}
