package ing.emojify.app.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FeelingsTableTest {
    @Test
    fun `resolveFeeling falls back to Neutral for unknown names`() {
        val resolved = resolveFeeling("NotARealFeeling")
        assertEquals(FEELINGS.getValue("Neutral"), resolved)
    }

    @Test
    fun `every style in FEELINGS has positive durations`() {
        FEELINGS.values.forEach {
            assertTrue(it.entranceMs > 0)
            assertTrue(it.emojiMs > 0)
        }
    }
}
