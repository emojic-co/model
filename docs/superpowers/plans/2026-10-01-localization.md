# Localization (English + Hebrew) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Localize the web and Android apps to English and Hebrew from one shared strings file, with a language selector in the web footer (cookie) and in Android Settings.

**Architecture:** `web/src/i18n.yml` (flat dotted keys under `en:` / `he:`) is the source of truth. `tools/data/export-style.ts` embeds it as an `i18n:` section of `style.yml`, which Android already bundles and OTA-refreshes. Web reads the yml directly via `?raw` (like `clip.yml`); Android parses the `i18n` section of `StyleFile` and exposes it through a `LocalStrings` CompositionLocal.

**Tech Stack:** React 19 + vitest (web), Bun + `yaml` (tooling), Kotlin + Jetpack Compose + kaml + JUnit (Android).

**Spec:** `docs/superpowers/specs/2026-10-01-localization-design.md`

## Global Constraints

- Languages: exactly `en` and `he`. Default = saved choice → device/browser language (`he*`/`iw` → `he`) → `en`.
- Never localized: the literals `jpg`, `gif`, `mp4`, `emojify.ing`, even inside translated sentences (e.g. `Copy card as jpg` → `העתקת הכרטיס כ־jpg`).
- Web language persisted in cookie `lang` (`path=/; max-age=31536000; SameSite=Lax`). Selector lives in the footer. Android selector lives on the Settings screen.
- Feeling identifiers stay English internally; only display labels are translated. Keys are `feeling.<Identifier>`.
- Card text, model input, `privacy.html`/static pages, `index.html` meta tags and Android `app_name` are not localized.
- RTL: Hebrew sets `<html lang dir="rtl">` on web and RTL layout direction for the Android Settings screen. The main card stage stays LTR (text aligns by content direction).
- Keys are flat (`feeling.Joyful`, not a nested `feelings:` map) so Kotlin can decode `Map<String, Map<String, String>>`. Placeholders are `{name}`.
- Per `CLAUDE.md`: run `ruff` only if Python is touched (it is, one line in `files.py`); commit only after the user has OK'd committing.

## Review Focus

- Key missing in `he` → falls back to `en`, never blank/`undefined` (Task 2 test).
- Unsupported / garbage `lang` cookie (`fr`, empty, `;;`) → browser language, then English (Task 2 test).
- `navigator.language` = `he-IL` or legacy `iw` → Hebrew (Task 2 test).
- Cookies/`document`/`navigator` unavailable (node tests, blocked cookies) → no throw (Task 2 test).
- Placeholder not supplied, or value containing `{x}` → left intact, no recursive substitution (Task 2 test).
- Unknown/future feeling id → label falls back to the identifier (Task 2 and Task 4 tests).
- Cached OTA `style.yml` with no `i18n` section → bundled strings used (Task 5 step).

## File Structure

- Create `web/src/i18n.yml` — all strings, both languages.
- Create `web/src/i18n.js` — pure helpers (`translate`, `feelingLabel`, `detectLang`, `readCookieLang`, cookie string builder) + React `LangProvider`/`useI18n`.
- Create `web/src/i18n.test.js` — vitest for helpers + yml parity/coverage.
- Modify `files.ts`, `files.py` — add `I18N_YML`.
- Modify `tools/data/export-style.ts` + `.test.ts` — embed/verify `i18n`.
- Modify web: `main.jsx`, `App.jsx`, `components/{Card,KeyHints,FeelingBar,ColorBar}.jsx`, `hooks/useCardImage.js`, `styles.css`.
- Modify `web/public/style.yml`, `android/app/src/main/assets/style.yml` — regenerated.
- Create `android/.../model/I18n.kt` (`Strings`, `LocalStrings`), `LanguagePrefs.kt`; modify `StyleFile.kt`, `MainActivity.kt`, `ui/MainScreen.kt`, `ui/SettingsScreen.kt`, `ui/components/{Card,FeelingBar}.kt`.
- Create `android/app/src/test/java/ing/emojify/model/I18nTest.kt`.
- Modify `CLAUDE.md` (one index line).

---

### Task 1: Shared strings file + export into style.yml

**Files:**
- Create: `web/src/i18n.yml`
- Modify: `files.ts:32`, `files.py:55`, `tools/data/export-style.ts`, `tools/data/export-style.test.ts`, `CLAUDE.md`, `docs/superpowers/specs/2026-10-01-localization-design.md`
- Regenerated: `web/public/style.yml`, `android/app/src/main/assets/style.yml`

**Interfaces:**
- Produces: `I18N_YML` constant; `style.yml` top-level `i18n: { en: {key: string}, he: {key: string} }`; `buildStyleFile().i18n`.

- [ ] **Step 1: Create `web/src/i18n.yml`**

```yaml
# UI strings for web and Android (ground truth; embedded into style.yml by `bun run export-style`).
# Flat dotted keys; {name} placeholders. Never translate: jpg, gif, mp4, emojify.ing.
en:
  input.placeholder: "type at least 3 characters…"
  warn.tooShort: "text is too short — showing a default card"
  timing.ran: "model ran in {ms}"
  keys.aria: "Keyboard shortcuts"
  keys.emoji: "emoji"
  keys.color: "color"
  keys.feeling: "feeling"
  keys.copy: "copy image"
  keys.clear: "clear text"
  contrast.label: "fix low-contrast palettes"
  footer.modelUpdated: "model updated {date}"
  footer.about: "about this model"
  footer.coverage: "emoji coverage"
  footer.stylePreview: "style preview"
  footer.madeBy: "made with ❤️ by Gilad"
  footer.patternsBy: "background patterns by"
  footer.language: "language"
  card.copyJpg: "Copy card as jpg"
  card.saveGif: "Save card as gif"
  card.saveMp4: "Save card as mp4"
  card.cancel: "cancel"
  card.cancelAria: "Cancel export"
  card.making: "Making {format}"
  color.aria: "color {n}"
  toast.nothing: "nothing to copy yet"
  toast.copied: "copied to clipboard ✓"
  toast.copyFailed: "copy failed"
  toast.cancelled: "cancelled"
  toast.saved: "{format} saved ✓"
  toast.failed: "{format} failed"
  main.settings: "Settings"
  main.clearText: "Clear text"
  main.exportFailed: "Export failed"
  settings.title: "Settings"
  settings.back: "Back"
  settings.autoUpdate: "Auto-update model"
  settings.wifiOnly: "Wi-Fi only"
  settings.maxEmojis: "Max emojis shown: {n}"
  settings.colorCount: "Number of colors to show: {n}"
  settings.exportRes: "jpg / mp4 export resolution"
  settings.debounce: "Typing delay before card updates: {ms} ms"
  settings.statusPending: "Update available, waiting for Wi-Fi"
  settings.statusUpdated: "Model updated {date}"
  settings.statusBundled: "Using bundled model"
  settings.checkNow: "Check now"
  settings.checking: "Checking…"
  settings.rate: "Rate this app"
  settings.language: "Language"
  feeling.Joyful: "Joyful"
  feeling.Excited: "Excited"
  feeling.Hopeful: "Hopeful"
  feeling.Serene: "Serene"
  feeling.Tender: "Tender"
  feeling.Playful: "Playful"
  feeling.Whimsical: "Whimsical"
  feeling.Awed: "Awed"
  feeling.Earnest: "Earnest"
  feeling.Determined: "Determined"
  feeling.Proud: "Proud"
  feeling.Wistful: "Wistful"
  feeling.Melancholy: "Melancholy"
  feeling.Anxious: "Anxious"
  feeling.Tense: "Tense"
  feeling.Furious: "Furious"
  feeling.Irritated: "Irritated"
  feeling.Disgusted: "Disgusted"
  feeling.Startled: "Startled"
  feeling.Sarcastic: "Sarcastic"
  feeling.Deadpan: "Deadpan"
  feeling.Neutral: "Neutral"
he:
  input.placeholder: "הקלידו לפחות 3 תווים…"
  warn.tooShort: "הטקסט קצר מדי — מוצג כרטיס ברירת מחדל"
  timing.ran: "המודל רץ ב־{ms}"
  keys.aria: "קיצורי מקלדת"
  keys.emoji: "אימוג׳י"
  keys.color: "צבע"
  keys.feeling: "רגש"
  keys.copy: "העתקת תמונה"
  keys.clear: "ניקוי טקסט"
  contrast.label: "תיקון פלטות בניגודיות נמוכה"
  footer.modelUpdated: "המודל עודכן {date}"
  footer.about: "על המודל"
  footer.coverage: "כיסוי אימוג׳י"
  footer.stylePreview: "תצוגת סגנונות"
  footer.madeBy: "נעשה באהבה ❤️ על ידי גלעד"
  footer.patternsBy: "דפוסי רקע מאת"
  footer.language: "שפה"
  card.copyJpg: "העתקת הכרטיס כ־jpg"
  card.saveGif: "שמירת הכרטיס כ־gif"
  card.saveMp4: "שמירת הכרטיס כ־mp4"
  card.cancel: "ביטול"
  card.cancelAria: "ביטול ייצוא"
  card.making: "יוצר {format}"
  color.aria: "צבע {n}"
  toast.nothing: "אין עדיין מה להעתיק"
  toast.copied: "הועתק ללוח ✓"
  toast.copyFailed: "ההעתקה נכשלה"
  toast.cancelled: "בוטל"
  toast.saved: "{format} נשמר ✓"
  toast.failed: "{format} נכשל"
  main.settings: "הגדרות"
  main.clearText: "ניקוי טקסט"
  main.exportFailed: "הייצוא נכשל"
  settings.title: "הגדרות"
  settings.back: "חזרה"
  settings.autoUpdate: "עדכון אוטומטי של המודל"
  settings.wifiOnly: "Wi-Fi בלבד"
  settings.maxEmojis: "מקסימום אימוג׳ים להצגה: {n}"
  settings.colorCount: "מספר צבעים להצגה: {n}"
  settings.exportRes: "רזולוציית ייצוא jpg / mp4"
  settings.debounce: "השהיה לפני עדכון הכרטיס: {ms} ms"
  settings.statusPending: "קיים עדכון, ממתין ל־Wi-Fi"
  settings.statusUpdated: "המודל עודכן {date}"
  settings.statusBundled: "משתמש במודל המובנה"
  settings.checkNow: "בדיקה עכשיו"
  settings.checking: "בודק…"
  settings.rate: "דרגו את האפליקציה"
  settings.language: "שפה"
  feeling.Joyful: "שמח"
  feeling.Excited: "נלהב"
  feeling.Hopeful: "מלא תקווה"
  feeling.Serene: "שליו"
  feeling.Tender: "עדין"
  feeling.Playful: "שובב"
  feeling.Whimsical: "גחמני"
  feeling.Awed: "נפעם"
  feeling.Earnest: "כן ורציני"
  feeling.Determined: "נחוש"
  feeling.Proud: "גאה"
  feeling.Wistful: "מתגעגע"
  feeling.Melancholy: "מלנכולי"
  feeling.Anxious: "חרד"
  feeling.Tense: "מתוח"
  feeling.Furious: "זועם"
  feeling.Irritated: "מעוצבן"
  feeling.Disgusted: "נגעל"
  feeling.Startled: "נבהל"
  feeling.Sarcastic: "סרקסטי"
  feeling.Deadpan: "יבש"
  feeling.Neutral: "ניטרלי"
```

- [ ] **Step 2: Write the failing drift tests** — in `tools/data/export-style.test.ts`, add `i18n` to both comparisons and a new test:

In the first test change the two object literals to include `i18n: onDisk.i18n` / `i18n: fresh.i18n`. In the android test add `expect(android.i18n).toEqual(web.i18n)`. Append:

```ts
test("style.yml carries both languages with identical keys", () => {
  const built = buildStyleFile()
  expect(Object.keys(built.i18n).sort()).toEqual(["en", "he"])
  expect(Object.keys(built.i18n.he).sort()).toEqual(Object.keys(built.i18n.en).sort())
})
```

- [ ] **Step 3: Run to verify failure**

Run: `bun test tools/data/export-style.test.ts`
Expected: FAIL (`i18n` undefined).

- [ ] **Step 4: Implement**

`files.ts` after the `CLIP_YML` line: `export const I18N_YML = \`${WEB_SRC_DIR}/i18n.yml\``. `files.py` after `CLIP_YML`: `I18N_YML = WEB_SRC_DIR / "i18n.yml"`.

`tools/data/export-style.ts`: add `I18N_YML` to the `files.ts` import, and in `buildStyleFile()` return, after `clip:`:

```ts
    i18n: parse(readFileSync(I18N_YML, "utf-8")),
```

- [ ] **Step 5: Regenerate and verify**

Run: `bun run export-style && bun test tools/data/export-style.test.ts`
Expected: PASS; `git diff --stat` shows `web/public/style.yml` and the android asset changed (also `exportedAt`).

- [ ] **Step 6: Docs** — in `CLAUDE.md`, extend the `web/src/clip.yml` bullet's neighbor with one line: ``- `web/src/i18n.yml` — shared en/he UI strings (ground truth, embedded into `style.yml` by `export-style`; web reads it directly, Android via `LocalStrings`).`` In the spec, change "`feelings:` map" to "flat `feeling.<Id>` keys" and the Android RTL line to "RTL layout direction on the Settings screen; the card stage stays LTR".

- [ ] **Step 7: Commit**

```bash
git add web/src/i18n.yml files.ts files.py tools/data web/public/style.yml android/app/src/main/assets/style.yml CLAUDE.md docs
git commit -m "feat(i18n): shared en/he strings embedded in style.yml"
```

---

### Task 2: Web i18n module (pure helpers + provider)

**Files:**
- Create: `web/src/i18n.js`, `web/src/i18n.test.js`
- Modify: `web/src/main.jsx`

**Interfaces:**
- Consumes: `web/src/i18n.yml` (Task 1), `FEELINGS` from `./feelings`, `ensureScriptFontsLoaded`/`scriptForLang` from `./scriptFonts`.
- Produces (`i18n.js`): `LANGS = ['en','he']`, `LANG_NAMES = {en:'English', he:'עברית'}`, `translate(lang, key, vars?) → string`, `feelingLabel(lang, id) → string`, `readCookieLang(cookieStr) → 'en'|'he'|null`, `detectLang(cookieStr, navLang) → 'en'|'he'`, `langCookie(lang) → string`, `LangProvider({children})`, `useI18n() → { lang, setLang, t(key, vars?), feeling(id) }`.

- [ ] **Step 1: Write the failing tests** (`web/src/i18n.test.js`)

```js
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
```

- [ ] **Step 2: Run to verify failure**

Run (in `web/`): `npx vitest run src/i18n.test.js`
Expected: FAIL (cannot resolve `./i18n`).

- [ ] **Step 3: Implement `web/src/i18n.js`**

```js
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
```

If `ensureScriptFontsLoaded` throws in a non-browser env, it is only called inside the effect, so node tests are unaffected. Check `scriptForLang('he')` returns the Hebrew script key and `scriptForLang('en')` the Latin one (read `web/src/scriptFonts.js`); if the Latin call is not a no-op, guard with `if (lang !== 'en')`.

- [ ] **Step 4: Wrap the app** — `web/src/main.jsx`:

```jsx
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { LangProvider } from './i18n'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <LangProvider>
    <App />
  </LangProvider>,
)
```

- [ ] **Step 5: Run to verify pass**

Run (in `web/`): `npx vitest run`
Expected: all PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add web/src/i18n.js web/src/i18n.test.js web/src/main.jsx
git commit -m "feat(web): i18n helpers, cookie detection and LangProvider"
```

---

### Task 3: Web UI wiring + footer selector

**Files:**
- Modify: `web/src/App.jsx`, `web/src/components/{Card,KeyHints,FeelingBar,ColorBar}.jsx`, `web/src/hooks/useCardImage.js`, `web/src/styles.css`

**Interfaces:**
- Consumes: `useI18n()` from Task 2 (`{lang, setLang, t, feeling}`), `LANGS`, `LANG_NAMES`.

Name clash to avoid: `App.jsx` already has a `lang` state (card text language). Destructure the UI language as `uiLang`: `const { lang: uiLang, setLang: setUiLang, t } = useI18n()`.

- [ ] **Step 1: `App.jsx`** — import `useI18n, LANGS, LANG_NAMES` from `./i18n`; add the destructure at the top of `App()`. Replace literals:
  - `placeholder="type at least 3 characters…"` → `placeholder={t('input.placeholder')}`
  - warning text → `{t('warn.tooShort')}`
  - `model ran in {formatMs(scores?.ms)}` → `{t('timing.ran', { ms: formatMs(scores?.ms) })}`
  - `fix low-contrast palettes` → `{t('contrast.label')}`
  - `model updated <span>{formatDate(meta?.exported_at)}</span>` → `{t('footer.modelUpdated', { date: formatDate(meta?.exported_at) })}` (drop the inner span)
  - the three footer links → `t('footer.about')`, `t('footer.coverage')`, `t('footer.stylePreview')`
  - `made with ❤️ by Gilad` → `{t('footer.madeBy')}`
  - `background patterns by{' '}` → `{t('footer.patternsBy')}{' '}` (link text "Hero Patterns" stays)
  - Pass `t` into hooks: `useCardImage(cardData, showToast, t)` / `useCardExport(cardData, showToast, t)`.
  - Add a language selector as the first `<span>` of the second `.footer-col`:

```jsx
<span className="lang-select">
  <label>
    {t('footer.language')}{' '}
    <select value={uiLang} onChange={(e) => setUiLang(e.target.value)}>
      {LANGS.map((l) => (
        <option key={l} value={l}>
          {LANG_NAMES[l]}
        </option>
      ))}
    </select>
  </label>
</span>
```

- [ ] **Step 2: `Card.jsx`** — add `const { t } = useI18n()` in `Card`; replace `aria-label={\`Making ${exporting}\`}` → `{t('card.making', { format: exporting })}`, `"Cancel export"` → `t('card.cancelAria')`, `cancel` → `{t('card.cancel')}`, and the three aria labels → `t('card.copyJpg')`, `t('card.saveGif')`, `t('card.saveMp4')`. Button text `jpg`/`gif`/`mp4` unchanged.

- [ ] **Step 3: `KeyHints.jsx`** — use label keys:

```jsx
import { useI18n } from '../i18n'

const HINTS = [
  { keys: ['↑', '↓'], label: 'keys.emoji' },
  { keys: ['Alt', '↑', '↓'], label: 'keys.color' },
  { keys: ['Ctrl', '↑', '↓'], label: 'keys.feeling' },
  { keys: ['Enter'], label: 'keys.copy' },
  { keys: ['Esc'], label: 'keys.clear' },
]

export function KeyHints() {
  const { t } = useI18n()
  return (
    <aside className="keys" aria-label={t('keys.aria')}>
```
and `<dd>{t(h.label)}</dd>`.

- [ ] **Step 4: `ColorBar.jsx`** — `const { t } = useI18n()`; `aria-label={\`color ${i + 1}\`}` → `aria-label={t('color.aria', { n: i + 1 })}`.

- [ ] **Step 5: `FeelingBar.jsx`** — in `FeelingButton` show the translated label; in `FeelingBar` resolve the font for the UI language so Hebrew labels get a Hebrew font:

```jsx
import { useI18n } from '../i18n'
// FeelingButton: add `label` prop → <button …>{label}</button>
// FeelingBar:
const { lang, feeling: feelingLabel } = useI18n()
// …
const r = resolveFeeling(f, lang)
// …
<FeelingButton key={f} feeling={f} label={feelingLabel(f)} active={f === active} style={style} onPick={onPick} />
```
`onPick` still receives the English identifier.

- [ ] **Step 6: `useCardImage.js`** — change signatures `useCardImage(cardData, showToast, t)` and `useCardExport(cardData, showToast, t)`; replace toasts: `'nothing to copy yet'` → `t('toast.nothing')`, `'copied to clipboard ✓'` → `t('toast.copied')`, `'copy failed'` → `t('toast.copyFailed')`, `'cancelled'` → `t('toast.cancelled')`, `` `${format} saved ✓` `` → `t('toast.saved', { format })`, `` `${format} failed` `` → `t('toast.failed', { format })`. Add `t` to the `useCallback` dependency arrays.

- [ ] **Step 7: `styles.css`** — add:

```css
.footer .lang-select select {
  font: inherit;
  letter-spacing: inherit;
  color: var(--ink);
  background: transparent;
  border: 1px solid var(--line);
  border-radius: 0.4em;
  padding: 0.1em 0.3em;
  cursor: pointer;
}

/* Hebrew: the stage grid keeps its LTR structure; only text runs RTL. */
html[dir='rtl'] .stage { direction: ltr; }
html[dir='rtl'] .footer-col,
html[dir='rtl'] .keys dd,
html[dir='rtl'] .contrast-toggle,
html[dir='rtl'] .input-meta,
html[dir='rtl'] .toast { direction: rtl; }
```
Also change `.footer-col { text-align: left }` to `text-align: start`.

- [ ] **Step 8: Verify**

Run (in `web/`): `npx vitest run && npm run build`
Expected: tests pass, build succeeds. Then `grep -nE "aria-label=\"|placeholder=\"|showToast\('" web/src` to confirm no English literals remain besides `jpg`/`gif`/`mp4`/`emojify.ing`. Start `npm run dev`, switch the footer select to עברית, reload and confirm the cookie persists, `<html dir="rtl">` is set, and the layout is intact (report to the user if you cannot view it).

- [ ] **Step 9: Commit**

```bash
git add web
git commit -m "feat(web): localize UI, footer language selector with cookie"
```

---

### Task 4: Android strings model + language prefs

**Files:**
- Modify: `android/app/src/main/java/ing/emojify/model/StyleFile.kt`
- Create: `android/app/src/main/java/ing/emojify/model/I18n.kt`, `LanguagePrefs.kt`, `android/app/src/test/java/ing/emojify/model/I18nTest.kt`

**Interfaces:**
- Produces: `StyleFile.i18n: Map<String, Map<String, String>>?`; `class Strings(val lang: String, table: Map<String, Map<String, String>>)` with `t(key, vars: Map<String, Any> = emptyMap()): String`, `feeling(id): String`; `val LocalStrings: ProvidableCompositionLocal<Strings>`; `LanguagePrefs.{FILE, KEY_LANG, SUPPORTED, NAMES, resolve(saved: String?, deviceLang: String): String}`.

- [ ] **Step 1: Write the failing test** (`I18nTest.kt`)

```kotlin
package ing.emojify.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class I18nTest {
    private val table = parseStyleFile(File("src/main/assets/style.yml").readText()).i18n!!
    private val en = Strings("en", table)
    private val he = Strings("he", table)

    private fun placeholders(s: String) = Regex("\\{(\\w+)}").findAll(s).map { it.groupValues[1] }.sorted().toList()

    @Test
    fun `en and he have identical keys and placeholders`() {
        assertEquals(table.getValue("en").keys, table.getValue("he").keys)
        for ((k, v) in table.getValue("en")) assertEquals(k, placeholders(v), placeholders(table.getValue("he").getValue(k)))
    }

    @Test
    fun `every style has a feeling label in both languages`() {
        for (id in Styles.file.styles.keys) {
            assertTrue(id, table.getValue("en").containsKey("feeling.$id"))
            assertTrue(id, table.getValue("he").containsKey("feeling.$id"))
        }
    }

    @Test
    fun `substitutes placeholders and leaves unknown ones`() {
        assertEquals("Max emojis shown: 7", en.t("settings.maxEmojis", mapOf("n" to 7)))
        assertEquals("color {n}", en.t("color.aria"))
        assertEquals("{n} saved ✓", en.t("toast.saved", mapOf("format" to "{n}", "n" to "x")))
    }

    @Test
    fun `falls back to en then key, feeling falls back to id`() {
        assertEquals("no.such.key", he.t("no.such.key"))
        assertEquals("Brand New", he.feeling("Brand New"))
        assertEquals("שמח", he.feeling("Joyful"))
    }

    @Test
    fun `literals jpg gif mp4 are kept in hebrew`() {
        assertTrue(he.t("card.copyJpg").contains("jpg"))
        assertTrue(he.t("settings.exportRes").contains("jpg / mp4"))
    }

    @Test
    fun `language resolution prefers saved, then device, then en`() {
        assertEquals("en", LanguagePrefs.resolve("en", "he"))
        assertEquals("he", LanguagePrefs.resolve(null, "he"))
        assertEquals("he", LanguagePrefs.resolve(null, "iw"))
        assertEquals("en", LanguagePrefs.resolve("fr", "fr"))
        assertEquals("en", LanguagePrefs.resolve(null, "de"))
    }
}
```

Note: `Styles.file` is only initialised by `Styles.init`. If `Styles.file` is not usable in a unit test, replace that loop's source with `parseStyleFile(...).styles.keys` (the parsed file held in a `private val file` in the test).

- [ ] **Step 2: Run to verify failure**

Run (in `android/`): `./gradlew :app:testDebugUnitTest --tests 'ing.emojify.model.I18nTest'`
Expected: FAIL (compile error: `i18n`/`Strings` unresolved). If gradle cannot run offline, report that and rely on compile review.

- [ ] **Step 3: Implement**

`StyleFile.kt` — add to `StyleFile` after `fonts`:

```kotlin
    val i18n: Map<String, Map<String, String>>? = null,
```

`I18n.kt`:

```kotlin
package ing.emojify.model

import androidx.compose.runtime.staticCompositionLocalOf

// UI strings come from the `i18n` section of style.yml (ground truth: web/src/i18n.yml).
class Strings(val lang: String, private val table: Map<String, Map<String, String>>) {
    fun t(key: String, vars: Map<String, Any> = emptyMap()): String {
        val s = table[lang]?.get(key) ?: table["en"]?.get(key) ?: return key
        return Regex("\\{(\\w+)}").replace(s) { m -> vars[m.groupValues[1]]?.toString() ?: m.value }
    }

    fun feeling(id: String): String = table[lang]?.get("feeling.$id") ?: table["en"]?.get("feeling.$id") ?: id
}

val LocalStrings = staticCompositionLocalOf<Strings> { error("LocalStrings not provided") }
```

Kotlin's `Regex.replace` with a lambda does not re-scan replacement output, so braces in values are safe.

`LanguagePrefs.kt`:

```kotlin
package ing.emojify.model

object LanguagePrefs {
    const val FILE = "language"
    const val KEY_LANG = "lang"
    val SUPPORTED = listOf("en", "he")
    val NAMES = mapOf("en" to "English", "he" to "עברית")

    fun resolve(saved: String?, deviceLang: String): String = when {
        saved in SUPPORTED -> saved!!
        deviceLang == "he" || deviceLang == "iw" -> "he"
        else -> "en"
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run (in `android/`): `./gradlew :app:testDebugUnitTest`
Expected: PASS (all unit tests).

- [ ] **Step 5: Commit**

```bash
git add android/app/src
git commit -m "feat(android): Strings model, LocalStrings and LanguagePrefs"
```

---

### Task 5: Android UI wiring + Settings language selector

**Files:**
- Modify: `MainActivity.kt`, `ui/MainScreen.kt`, `ui/SettingsScreen.kt`, `ui/components/Card.kt`, `ui/components/FeelingBar.kt` (all under `android/app/src/main/java/ing/emojify/`)

**Interfaces:**
- Consumes: `Strings`, `LocalStrings`, `LanguagePrefs` (Task 4).
- Produces: `SettingsScreen(onBack: () -> Unit, lang: String, onLangChange: (String) -> Unit)`.

- [ ] **Step 1: `MainActivity.kt`** — keep the chosen style's strings, falling back to the bundled ones when a cached style lacks `i18n`:

```kotlin
val chosenStyle = if (cachedStyle != null && cachedStyle.exportedAt > bundledStyle.exportedAt) cachedStyle else bundledStyle
Styles.init(chosenStyle)
val i18n = (chosenStyle.i18n ?: bundledStyle.i18n)!!
val langPrefs = getSharedPreferences(LanguagePrefs.FILE, Context.MODE_PRIVATE)
```

Inside `setContent { MaterialTheme { Surface { … } } }`:

```kotlin
var lang by remember {
    mutableStateOf(LanguagePrefs.resolve(langPrefs.getString(LanguagePrefs.KEY_LANG, null), java.util.Locale.getDefault().language))
}
val strings = remember(lang) { Strings(lang, i18n) }
CompositionLocalProvider(LocalStrings provides strings) {
    var showSettings by remember { mutableStateOf(false) }
    BackHandler(enabled = showSettings) { showSettings = false }
    if (showSettings) {
        SettingsScreen(
            onBack = { showSettings = false },
            lang = lang,
            onLangChange = {
                lang = it
                langPrefs.edit().putString(LanguagePrefs.KEY_LANG, it).apply()
            },
        )
    } else {
        MainScreen(meta = meta, predictor = predictor, onSettingsClick = { showSettings = true })
    }
}
```
Add imports for `CompositionLocalProvider`, `LanguagePrefs`, `Strings`, `LocalStrings`.

- [ ] **Step 2: `MainScreen.kt`** — add `val strings = LocalStrings.current` at the top of `MainScreen`; replace `"type at least 3 characters…"` → `strings.t("input.placeholder")`, `"Export failed"` → `strings.t("main.exportFailed")`, `contentDescription = "Clear text"` → `strings.t("main.clearText")`, `contentDescription = "Settings"` → `strings.t("main.settings")`, `"made with ❤️ by Gilad"` → `strings.t("footer.madeBy")`. Keep the existing `lang` (card text language) untouched.

- [ ] **Step 3: `Card.kt`** — the cancel button: `ShareButton(label = "cancel", …)` → `ShareButton(label = LocalStrings.current.t("card.cancel"), …)`. `ShareFormat` labels (`jpg`/`gif`/`mp4`) are unchanged.

- [ ] **Step 4: `FeelingBar.kt`** — in `FeelingSwatch` use the UI language for font and label:

```kotlin
val strings = LocalStrings.current
val style = resolveFeeling(feeling, strings.lang)
// …
val label = strings.feeling(feeling)
val displayText = if (style.uppercase) label.uppercase() else label
```
`onPick(feeling)` still passes the English identifier. If any `contentDescription`/semantics uses `feeling`, use `label`.

- [ ] **Step 5: `SettingsScreen.kt`** — change the signature and add `val strings = LocalStrings.current`. Make `statusText` take `strings`:

```kotlin
private fun statusText(updater: ModelUpdater, strings: Strings): String {
    val installed = updater.installedVersion()
    return when {
        updater.pendingVersion() != null -> strings.t("settings.statusPending")
        installed != null -> strings.t("settings.statusUpdated", mapOf("date" to installed.substringBefore("T")))
        else -> strings.t("settings.statusBundled")
    }
}
```
Update both `statusText(updater)` call sites to `statusText(updater, strings)`. Literal replacements:

| Before | After |
|---|---|
| `contentDescription = "Back"` | `strings.t("settings.back")` |
| `Text("Settings", …)` | `Text(strings.t("settings.title"), …)` |
| `"Auto-update model"` | `strings.t("settings.autoUpdate")` |
| `"Wi-Fi only"` | `strings.t("settings.wifiOnly")` |
| `"Max emojis shown: $maxEmojis"` | `strings.t("settings.maxEmojis", mapOf("n" to maxEmojis))` |
| `"Number of colors to show: $colorCount"` | `strings.t("settings.colorCount", mapOf("n" to colorCount))` |
| `"jpg / mp4 export resolution"` | `strings.t("settings.exportRes")` |
| `"Typing delay before card updates: $debounceMs ms"` | `strings.t("settings.debounce", mapOf("ms" to debounceMs))` |
| `if (checking) "Checking…" else "Check now"` | `strings.t(if (checking) "settings.checking" else "settings.checkNow")` |
| `"Rate this app"` | `strings.t("settings.rate")` |

Add the language row after the debounce slider block (before the status `Text`), mirroring the export-size selector:

```kotlin
Spacer(modifier = Modifier.height(24.dp))
Text(strings.t("settings.language"))
Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
    for (code in LanguagePrefs.SUPPORTED) {
        val name = LanguagePrefs.NAMES.getValue(code)
        if (code == lang) Button(onClick = { onLangChange(code) }) { Text(name) }
        else OutlinedButton(onClick = { onLangChange(code) }) { Text(name) }
    }
}
```

Wrap the screen body in RTL for Hebrew: make the existing `Box(…) { … }` the content of

```kotlin
CompositionLocalProvider(
    LocalLayoutDirection provides if (lang == "he") LayoutDirection.Rtl else LayoutDirection.Ltr,
) { /* existing Box */ }
```
with imports `CompositionLocalProvider`, `LocalLayoutDirection` (`androidx.compose.ui.platform`), `LayoutDirection` (`androidx.compose.ui.unit`), `ing.emojify.model.{LanguagePrefs, LocalStrings, Strings}`. In RTL the back arrow is already `AutoMirrored`.

- [ ] **Step 6: Verify**

Run (in `android/`): `./gradlew :app:testDebugUnitTest :app:compileDebugKotlin`
Expected: PASS / BUILD SUCCESSFUL. Then `grep -nE 'Text\("[A-Za-z]|contentDescription = "' android/app/src/main/java` and confirm no English UI literals remain (exceptions: `"Aa"` swatch, share-format labels, and `emojify.ing`). Android Compose output cannot be viewed from here; say so in the report.

- [ ] **Step 6b: Push to the phone** (standing user request: install after every Android change)

Run (in `android/`): `./gradlew installDebug` (phone is on adb: `adb devices`). Report success/failure; if no device is listed, say so.

- [ ] **Step 7: Commit**

```bash
git add android
git commit -m "feat(android): localized UI and Settings language selector"
```

---

### Task 6: Full verification

- [ ] **Step 1:** `bun run export-style && bun test tools/data/export-style.test.ts` — PASS, and `git status` shows no uncommitted regenerated files (i.e. the committed `style.yml` copies are current).
- [ ] **Step 2:** `cd web && npx vitest run && npm run build` — PASS.
- [ ] **Step 3:** `cd android && ./gradlew :app:testDebugUnitTest installDebug` — PASS and installed on the phone (or report exactly what could not run).
- [ ] **Step 4:** `ruff check files.py` — clean.
- [ ] **Step 5:** Report outcome honestly: what was verified by tests/build, and that Hebrew layout on both platforms was not visually checked unless it was.
