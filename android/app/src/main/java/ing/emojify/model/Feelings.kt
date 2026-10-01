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
    // The face actually requested from the font provider (web loads exactly these from Google Fonts).
    val faceWeight: Int = 400,
    val faceItalic: Boolean = false,
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

private fun StyleEntry.toFeelingStyle(patterns: Map<String, String>) = FeelingStyle(
    cluster = cluster,
    fontName = font,
    fontWeight = fontWeight,
    italic = italic,
    uppercase = uppercase,
    letterSpacingEm = letterSpacingEm,
    faceWeight = fontWeight,
    faceItalic = italic,
    entranceMs = entranceMs,
    emojiMs = emojiMs,
    entranceMotif = entrance,
    emojiMotif = emoji,
    patternSvg = patterns.getValue(pattern.name),
    patternWidthRatio = pattern.widthRatio,
    patternHeightRatio = pattern.heightRatio,
)

fun resolveFeeling(feeling: String?, lang: String? = "en"): FeelingStyle {
    val entry = Styles.file.styles[feeling] ?: Styles.file.styles.getValue("Neutral")
    val base = entry.toFeelingStyle(Styles.file.patterns)
    val script = scriptForLang(lang)
    if (script == LATIN) return base
    val face = fontSpecForScript(script, base.cluster)
    return base.copy(fontName = face?.family ?: FALLBACK_FONT_FAMILY, faceWeight = face?.weight ?: 400, faceItalic = face?.italic ?: false)
}

/** The string a card shows: upper-cased for uppercase styles, with the placeholder when blank. */
fun cardDisplayText(text: String, style: FeelingStyle): String =
    (if (style.uppercase) text.uppercase() else text).ifBlank { "What's on your mind?" }
