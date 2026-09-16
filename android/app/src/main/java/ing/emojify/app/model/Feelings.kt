package ing.emojify.app.model

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
)

val FEELINGS: Map<String, FeelingStyle> = mapOf(
    "Joyful" to FeelingStyle("joy", "Fredoka", bold = true, entranceMs = 560, emojiMs = 900),
    "Excited" to FeelingStyle("joy", "Chewy", uppercase = true, letterSpacingEm = 0.05f, entranceMs = 460, emojiMs = 380),
    "Hopeful" to FeelingStyle("drive", "Poppins", entranceMs = 780, emojiMs = 3000),
    "Serene" to FeelingStyle("calm", "Quicksand", entranceMs = 900, emojiMs = 4200),
    "Tender" to FeelingStyle("tender", "Caveat", bold = true, entranceMs = 700, emojiMs = 1300),
    "Playful" to FeelingStyle("play", "Bungee", entranceMs = 600, emojiMs = 1100),
    "Whimsical" to FeelingStyle("play", "Gochi Hand", letterSpacingEm = 0.02f, entranceMs = 640, emojiMs = 1500),
    "Awed" to FeelingStyle("reflective", "Luckiest Guy", letterSpacingEm = 0.04f, entranceMs = 520, emojiMs = 2600),
    "Earnest" to FeelingStyle("tender", "Shadows Into Light", letterSpacingEm = 0.01f, entranceMs = 720, emojiMs = 1600),
    "Determined" to FeelingStyle("drive", "Barlow Condensed", uppercase = true, bold = true, entranceMs = 560, emojiMs = 1400),
    "Proud" to FeelingStyle("drive", "Rubik", uppercase = true, bold = true, letterSpacingEm = 0.05f, entranceMs = 700, emojiMs = 2600),
    "Wistful" to FeelingStyle("sad", "Spectral", italic = true, letterSpacingEm = 0.05f, entranceMs = 1050, emojiMs = 4200),
    "Melancholy" to FeelingStyle("sad", "Playfair Display", italic = true, entranceMs = 1000, emojiMs = 3200),
    "Anxious" to FeelingStyle("anxiety", "Shantell Sans", entranceMs = 560, emojiMs = 220),
    "Tense" to FeelingStyle("anxiety", "Oswald", letterSpacingEm = -0.01f, entranceMs = 500, emojiMs = 420),
    "Furious" to FeelingStyle("anger", "Anton", uppercase = true, letterSpacingEm = 0.06f, entranceMs = 420, emojiMs = 450),
    "Irritated" to FeelingStyle("anger", "Archivo Black", uppercase = true, entranceMs = 520, emojiMs = 600),
    "Disgusted" to FeelingStyle("anger", "Griffy", italic = true, letterSpacingEm = 0.03f, entranceMs = 480, emojiMs = 700),
    "Startled" to FeelingStyle("play", "Schoolbell", entranceMs = 420, emojiMs = 2600),
    "Sarcastic" to FeelingStyle("reflective", "Bitter", italic = true, entranceMs = 800, emojiMs = 4200),
    "Deadpan" to FeelingStyle("reflective", "Inter", entranceMs = 700, emojiMs = 6000),
    "Neutral" to FeelingStyle("reflective", "Work Sans", bold = true, entranceMs = 650, emojiMs = 3200),
)

fun resolveFeeling(feeling: String?): FeelingStyle = FEELINGS[feeling] ?: FEELINGS.getValue("Neutral")
