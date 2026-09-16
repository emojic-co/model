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
    val bold: Boolean = false,
    val italic: Boolean = false,
    val uppercase: Boolean = false,
    val letterSpacingEm: Float? = null,
    val entranceMs: Int,
    val emojiMs: Int,
    val entranceMotif: String,
    val emojiMotif: String,
)

val FEELINGS: Map<String, FeelingStyle> = mapOf(
    "Joyful" to FeelingStyle("joy", "Fredoka", bold = true, entranceMs = 560, emojiMs = 900, entranceMotif = "pop", emojiMotif = "hop"),
    "Excited" to FeelingStyle("joy", "Chewy", uppercase = true, letterSpacingEm = 0.05f, entranceMs = 460, emojiMs = 380, entranceMotif = "pop", emojiMotif = "hop"),
    "Hopeful" to FeelingStyle("drive", "Poppins", entranceMs = 780, emojiMs = 3000, entranceMotif = "rise", emojiMotif = "lift"),
    "Serene" to FeelingStyle("calm", "Quicksand", entranceMs = 900, emojiMs = 4200, entranceMotif = "settle", emojiMotif = "breathe"),
    "Tender" to FeelingStyle("tender", "Caveat", bold = true, entranceMs = 700, emojiMs = 1300, entranceMotif = "bloom", emojiMotif = "heartbeat"),
    "Playful" to FeelingStyle("play", "Bungee", entranceMs = 600, emojiMs = 1100, entranceMotif = "spin", emojiMotif = "wobble"),
    "Whimsical" to FeelingStyle("play", "Gochi Hand", letterSpacingEm = 0.02f, entranceMs = 640, emojiMs = 1500, entranceMotif = "spin", emojiMotif = "wobble"),
    "Awed" to FeelingStyle("reflective", "Luckiest Guy", letterSpacingEm = 0.04f, entranceMs = 520, emojiMs = 2600, entranceMotif = "fadeTilt", emojiMotif = "tilt"),
    "Earnest" to FeelingStyle("tender", "Shadows Into Light", letterSpacingEm = 0.01f, entranceMs = 720, emojiMs = 1600, entranceMotif = "bloom", emojiMotif = "heartbeat"),
    "Determined" to FeelingStyle("drive", "Barlow Condensed", uppercase = true, bold = true, entranceMs = 560, emojiMs = 1400, entranceMotif = "rise", emojiMotif = "lift"),
    "Proud" to FeelingStyle("drive", "Rubik", uppercase = true, bold = true, letterSpacingEm = 0.05f, entranceMs = 700, emojiMs = 2600, entranceMotif = "rise", emojiMotif = "lift"),
    "Wistful" to FeelingStyle("sad", "Spectral", italic = true, letterSpacingEm = 0.05f, entranceMs = 1050, emojiMs = 4200, entranceMotif = "drop", emojiMotif = "sink"),
    "Melancholy" to FeelingStyle("sad", "Playfair Display", italic = true, entranceMs = 1000, emojiMs = 3200, entranceMotif = "drop", emojiMotif = "sink"),
    "Anxious" to FeelingStyle("anxiety", "Shantell Sans", entranceMs = 560, emojiMs = 220, entranceMotif = "jitter", emojiMotif = "tremor"),
    "Tense" to FeelingStyle("anxiety", "Oswald", letterSpacingEm = -0.01f, entranceMs = 500, emojiMs = 420, entranceMotif = "jitter", emojiMotif = "tremor"),
    "Furious" to FeelingStyle("anger", "Anton", uppercase = true, letterSpacingEm = 0.06f, entranceMs = 420, emojiMs = 450, entranceMotif = "slam", emojiMotif = "shake"),
    "Irritated" to FeelingStyle("anger", "Archivo Black", uppercase = true, entranceMs = 520, emojiMs = 600, entranceMotif = "slam", emojiMotif = "shake"),
    "Disgusted" to FeelingStyle("anger", "Griffy", italic = true, letterSpacingEm = 0.03f, entranceMs = 480, emojiMs = 700, entranceMotif = "slam", emojiMotif = "shake"),
    "Startled" to FeelingStyle("play", "Schoolbell", entranceMs = 420, emojiMs = 2600, entranceMotif = "shrinkBack", emojiMotif = "shrinkBack"),
    "Sarcastic" to FeelingStyle("reflective", "Bitter", italic = true, entranceMs = 800, emojiMs = 4200, entranceMotif = "fadeTilt", emojiMotif = "tilt"),
    "Deadpan" to FeelingStyle("reflective", "Inter", entranceMs = 700, emojiMs = 6000, entranceMotif = "droop", emojiMotif = "droop"),
    "Neutral" to FeelingStyle("reflective", "Work Sans", bold = true, entranceMs = 650, emojiMs = 3200, entranceMotif = "fadeTilt", emojiMotif = "tilt"),
)

fun resolveFeeling(feeling: String?, lang: String? = "en"): FeelingStyle {
    val base = FEELINGS[feeling] ?: FEELINGS.getValue("Neutral")
    val script = scriptForLang(lang)
    return if (script == LATIN) base else base.copy(fontName = fontForScript(script, base.cluster))
}
