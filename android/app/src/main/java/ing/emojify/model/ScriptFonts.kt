package ing.emojify.model

const val LATIN = "latin"

private val LANG_SCRIPT = mapOf(
    "ar" to "arabic",
    "bg" to "cyrillic",
    "el" to "greek",
    "he" to "hebrew",
    "hi" to "devanagari",
    "mr" to "devanagari",
    "ja" to "japanese",
    "ko" to "korean",
    "ru" to "cyrillic",
    "th" to "thai",
    "uk" to "cyrillic",
    "zh" to "chinese",
)

fun scriptForLang(lang: String?): String = LANG_SCRIPT[lang] ?: LATIN

private val HEBREW_RE = Regex("[\\u0590-\\u05FF]")

fun langForText(text: String): String = if (HEBREW_RE.containsMatchIn(text)) "he" else "en"

/** Font family for a script's cluster, from the shared fonts.yml (via style.yml). */
fun fontForScript(script: String, cluster: String): String {
    val fonts = Styles.file.fonts ?: return FALLBACK_FONT
    return (fonts.scripts[script]?.get(cluster) ?: fonts.fallback).family
}

// Used only while a cached style.yml predates the `fonts` section.
private const val FALLBACK_FONT = "Noto Sans"
