package ing.emojify.model

import org.junit.Assert.assertEquals
import org.junit.Test

class FeelingsTest {
    @Test
    fun `topFeelings ranks by score descending`() {
        val scores = floatArrayOf(0.1f, 0.9f, 0.5f)
        val feelings = listOf("Joyful", "Excited", "Hopeful")
        assertEquals(listOf("Excited", "Hopeful", "Joyful"), topFeelings(scores, feelings, null, count = 3))
    }

    @Test
    fun `topFeelings keeps the selected feeling even if outside top-N`() {
        val scores = floatArrayOf(0.1f, 0.9f, 0.5f, 0.05f)
        val feelings = listOf("Joyful", "Excited", "Hopeful", "Serene")
        val result = topFeelings(scores, feelings, "Serene", count = 2)
        assertEquals(2, result.size)
        assertEquals("Excited", result[0])
        assertEquals("Serene", result[1])
    }

    @Test
    fun `topFeelings returns empty list when scores are null`() {
        assertEquals(emptyList<String>(), topFeelings(null, listOf("Joyful"), null, count = 3))
    }
}
