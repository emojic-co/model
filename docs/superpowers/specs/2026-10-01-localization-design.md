# Localization (English + Hebrew) — design

## Goal
Web (`web/`) and Android (`android/`) UIs are localized to English and Hebrew from a single strings file, so both stay in sync. Android gets a language selector in Settings; web gets one in the footer, persisted in a cookie.

## Scope
- In: all UI chrome (placeholder, warnings, buttons, aria labels, toasts, key hints, footer, Android settings, content descriptions) and display names of the 21 feelings plus `Neutral`.
- Never localized: the literals `jpg`, `gif`, `mp4` and `emojify.ing` (including inside otherwise-translated strings, e.g. `Copy card as jpg`, which keep the literal as-is, or are passed in as placeholders).
- Out: card text, model input, static pages (`privacy.html`, etc.), `index.html` title/meta tags, Android `app_name`, Play listing, other languages.

## Shared strings
- `web/src/i18n.yml` is the source of truth: top-level `en:` and `he:` with identical key sets, including flat `feeling.<Id>` keys (display labels; identifiers stay English internally).
- Placeholders use `{name}` syntax (e.g. `Max emojis shown: {n}`).
- `tools/data/export-style.ts` embeds the file as an `i18n:` section of `style.yml` (same mechanism as `clip`). `style.yml` is already bundled in Android and OTA-refreshed by `StyleUpdater.kt`; `export-style.test.ts` fails on drift.
- A test asserts `en`/`he` key and placeholder parity.

## Web
- `web/src/i18n.js`: `t(key, vars)`, `useLang()`/context, cookie helpers.
- Language resolution: `lang` cookie (`SameSite=Lax`, 1 year) → `navigator.language` (`he*` → Hebrew) → English.
- Setting the language updates `<html lang>` and `dir` (`rtl` for Hebrew).
- Footer gets a `<select>` with `English` / `עברית`.
- Literals in `App.jsx`, `Card`, `KeyHints`, `FeelingBar`, `ColorBar`, `Toast`, `useCardImage` move to `t()`; the feeling bar/card show translated feeling names.

## Android
- `I18n.kt` parses the `i18n:` section via the existing `StyleFile`/kaml path; `LocalStrings` CompositionLocal exposes `t(key, vars)`.
- `LanguagePrefs` (same pattern as other `*Prefs`): default device locale (`he` → Hebrew, else English); explicit choice overrides. Language is state in `MainActivity`, so changes recompose without restart.
- RTL layout direction on the Settings screen for Hebrew; the card stage stays LTR.
- Literals in `MainScreen`, `SettingsScreen`, `FeelingBar` etc. move to `t()`; Settings gets a language row of `English`/`עברית` buttons styled like the export-size selector.
- Cached OTA `style.yml` lacking `i18n` falls back to the bundled one.

## Testing
- vitest: `t()` + placeholders, parity, cookie/detection logic.
- Drift test for the `i18n` section in `export-style.test.ts`.
- Android JUnit: parity/placeholders on the parsed section, locale-default logic.
- Run `bun run export-style`, `npm test`, `npm run build`; Android verified by compile + unit tests only (no emulator).
