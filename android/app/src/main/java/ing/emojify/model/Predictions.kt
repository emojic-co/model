package ing.emojify.model

data class EmojiScore(val emoji: String, val p: Float)

fun pickEmojiList(emojiLogits: FloatArray, emojis: List<String>, slots: Int): List<EmojiScore> {
    val sigmoids = sigmoid(emojiLogits)
    return emojiLogits.indices
        .sortedByDescending { emojiLogits[it] }
        .take(slots)
        .map { EmojiScore(emojis[it], sigmoids[it]) }
}
