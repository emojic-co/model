const TRANSLATABLE_LANGS = [
  'ar', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'he', 'hi', 'hr', 'hu',
  'id', 'it', 'ja', 'kn', 'ko', 'lt', 'mr', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl',
  'sv', 'ta', 'te', 'th', 'tr', 'uk', 'vi', 'zh', 'zh-Hant',
]

export function translationSupported() {
  return typeof self !== 'undefined' && 'Translator' in self && 'LanguageDetector' in self
}

export function languageName(code) {
  if (!code) return code
  if (typeof Intl === 'undefined' || !Intl.DisplayNames) return code
  try {
    return new Intl.DisplayNames(['en'], { type: 'language' }).of(code) ?? code
  } catch {
    return code
  }
}

export function supportedLanguages() {
  return TRANSLATABLE_LANGS.filter((code) => code !== 'en')
    .map((code) => ({ code, name: languageName(code) }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

let detectorPromise = null
function getDetector() {
  if (!detectorPromise) {
    detectorPromise = LanguageDetector.create().catch((err) => {
      detectorPromise = null
      throw err
    })
  }
  return detectorPromise
}

export function warmupDetector() {
  if (!translationSupported()) return
  getDetector().catch((err) => console.warn('emojic: language detector warmup failed', err))
}

const translators = new Map()
function getTranslator(sourceLanguage) {
  let p = translators.get(sourceLanguage)
  if (!p) {
    p = Translator.create({ sourceLanguage, targetLanguage: 'en' }).catch((err) => {
      translators.delete(sourceLanguage)
      throw err
    })
    translators.set(sourceLanguage, p)
  }
  return p
}

async function detectLanguage(text) {
  try {
    const detector = await getDetector()
    const [best] = await detector.detect(text)
    const lang = best?.detectedLanguage
    return !lang || lang === 'und' ? null : lang
  } catch (err) {
    console.warn('emojic: language detection unavailable', err)
    return null
  }
}

export async function detectAndTranslate(text, forcedLang = null) {
  if (!translationSupported()) {
    return { text, lang: 'en', detectedLang: null, translated: false, unsupported: true }
  }

  const detectedLang = await detectLanguage(text)
  const lang = forcedLang ?? detectedLang ?? 'en'

  if (lang === 'en') {
    return { text, lang: 'en', detectedLang, translated: false, unsupported: false }
  }

  try {
    const availability = await Translator.availability({ sourceLanguage: lang, targetLanguage: 'en' })
    if (availability === 'unavailable') {
      return { text, lang, detectedLang, translated: false, unsupported: false }
    }
    const translator = await getTranslator(lang)
    const translated = await translator.translate(text)
    return { text: translated, lang, detectedLang, translated: true, unsupported: false }
  } catch (err) {
    console.warn(`emojic: translation from "${lang}" failed, running original text through the model`, err)
    return { text, lang, detectedLang, translated: false, unsupported: false }
  }
}
