package ing.emojify.model

fun topFeelings(
    feelingScores: FloatArray?,
    feelings: List<String>,
    selected: String?,
    count: Int,
): List<String> {
    if (feelingScores == null) return emptyList()
    val ranked = feelings.indices.sortedByDescending { feelingScores[it] }.map { feelings[it] }
    val top = ranked.take(count)
    return if (selected != null && selected !in top) ranked.take(count - 1) + selected else top
}

data class FeelingStyle(
    val cluster: String,
    val fontName: String,
    val fontWeight: Int = 400,
    val italic: Boolean = false,
    val uppercase: Boolean = false,
    val letterSpacingEm: Float? = null,
    val entranceMs: Int,
    val emojiMs: Int,
    val entranceMotif: String,
    val emojiMotif: String,
    val patternSvg: String,
    val patternWidthRatio: Float,
    val patternHeightRatio: Float,
)

object Styles {
    lateinit var file: StyleFile
        private set

    fun init(file: StyleFile) {
        this.file = file
    }
}

private fun StyleEntry.toFeelingStyle() = FeelingStyle(
    cluster = cluster,
    fontName = font,
    fontWeight = fontWeight,
    italic = italic,
    uppercase = uppercase,
    letterSpacingEm = letterSpacingEm,
    entranceMs = entranceMs,
    emojiMs = emojiMs,
    entranceMotif = entrance,
    emojiMotif = emoji,
    patternSvg = pattern.svg,
    patternWidthRatio = pattern.widthRatio,
    patternHeightRatio = pattern.heightRatio,
)

fun resolveFeeling(feeling: String?, lang: String? = "en"): FeelingStyle {
    val entry = Styles.file.styles[feeling] ?: Styles.file.styles.getValue("Neutral")
    val base = entry.toFeelingStyle()
    val script = scriptForLang(lang)
    return if (script == LATIN) base else base.copy(fontName = fontForScript(script, base.cluster))
}
