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
