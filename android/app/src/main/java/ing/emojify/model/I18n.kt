package ing.emojify.model

import androidx.compose.runtime.staticCompositionLocalOf

// UI strings come from the `i18n` section of style.yml (ground truth: web/src/i18n.yml).
class Strings(val lang: String, private val table: Map<String, Map<String, String>>) {
    fun t(key: String, vars: Map<String, Any> = emptyMap()): String {
        val s = table[lang]?.get(key) ?: table["en"]?.get(key) ?: return key
        return Regex("\\{(\\w+)\\}").replace(s) { m -> vars[m.groupValues[1]]?.toString() ?: m.value }
    }

    fun feeling(id: String): String = table[lang]?.get("feeling.$id") ?: table["en"]?.get("feeling.$id") ?: id
}

val LocalStrings = staticCompositionLocalOf<Strings> { error("LocalStrings not provided") }
