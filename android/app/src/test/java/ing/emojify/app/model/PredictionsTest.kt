package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Test

class PredictionsTest {
    @Test
    fun `pickEmojiList returns top-N by logit, sigmoid-scored`() {
        val logits = floatArrayOf(0.1f, 5.0f, -2.0f, 1.0f)
        val emojis = listOf("🙂", "🚗", "🌊", "📅")
        val top = pickEmojiList(logits, emojis, slots = 2)
        assertEquals(listOf("🚗", "📅"), top.map { it.emoji })
        assertEquals(sigmoid(floatArrayOf(5.0f))[0], top[0].p, 0.0001f)
    }
}
