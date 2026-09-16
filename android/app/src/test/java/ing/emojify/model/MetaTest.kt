package ing.emojify.model

import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Test

class MetaTest {
    private val sampleJson = """
        {
          "chars": "·ab",
          "pad_idx": 0,
          "max_text_len": 4,
          "emojis": ["🙂", "🚗"],
          "styles": ["Joyful", "Tense"]
        }
    """.trimIndent()

    @Test
    fun `parses meta json`() {
        val meta = Json.decodeFromString(Meta.serializer(), sampleJson)
        assertEquals("·ab", meta.chars)
        assertEquals(0, meta.pad_idx)
        assertEquals(4, meta.max_text_len)
        assertEquals(listOf("🙂", "🚗"), meta.emojis)
        assertEquals(listOf("Joyful", "Tense"), meta.styles)
    }

    @Test
    fun `builds char to index map in declared order`() {
        val meta = Json.decodeFromString(Meta.serializer(), sampleJson)
        val map = meta.charToIndex()
        assertEquals(0, map['·'])
        assertEquals(1, map['a'])
        assertEquals(2, map['b'])
    }
}
