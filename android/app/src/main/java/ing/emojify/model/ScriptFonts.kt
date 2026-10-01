package ing.emojify.model

const val LATIN = "latin"

private val LANG_SCRIPT = mapOf(
    "he" to "hebrew",
)

fun scriptForLang(lang: String?): String = LANG_SCRIPT[lang] ?: LATIN

private val HEBREW_RE = Regex("[\\u0590-\\u05FF]")

fun langForText(text: String): String = if (HEBREW_RE.containsMatchIn(text)) "he" else "en"

/** Font for a script's cluster, from the shared fonts.yml (via style.yml); null while a cached style.yml predates it. */
fun fontSpecForScript(script: String, cluster: String): FontSpec? {
    val fonts = Styles.file.fonts ?: return null
    return fonts.scripts[script]?.get(cluster) ?: fonts.fallback
}

// Used only while a cached style.yml predates the `fonts` section.
const val FALLBACK_FONT_FAMILY = "Noto Sans"
