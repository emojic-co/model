package ing.emojify.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ModelIoTest {
    private val charSet = "·abcdefghijklmnopqrstuvwxyz0123456789!?:()@$%&* ".toSet()

    @Test
    fun `normalize lowercases, collapses whitespace, drops unknown chars, caps repeats at 2`() {
        assertEquals("hi there", normalize("  Hi   there", charSet))
        assertEquals("aab", normalize("aaaab", charSet))
        assertEquals("ab", normalize("aéb", charSet)) // é is not in charSet, dropped
    }

    @Test
    fun `encode pads to max_text_len with pad_idx and truncates`() {
        val meta = Meta(chars = "·ab", pad_idx = 0, max_text_len = 4, emojis = emptyList(), styles = emptyList())
        val char2idx = meta.charToIndex()
        val ids = encode("aabbb", meta, char2idx)
        assertEquals(4, ids.size)
        assertEquals(1L, ids[0]) // 'a'
        assertEquals(1L, ids[1]) // 'a'
        assertEquals(2L, ids[2]) // 'b'
        assertEquals(2L, ids[3]) // 'b' (truncated at max_text_len=4, "aabb" after collapse of 3rd b)
    }

    @Test
    fun `decodeColorList interprets floats as OKLAB-unit, not raw RGB bytes`() {
        // unit (1,0,0) -> white; unit (-1,0,0) -> black; unit (1,0,0) again -> white
        val flat = floatArrayOf(1f, 0f, 0f, -1f, 0f, 0f, 1f, 0f, 0f)
        val palettes = decodeColorList(flat)
        assertEquals(1, palettes.size)
        assertEquals("#ffffff", palettes[0].bg1)
        assertEquals("#000000", palettes[0].bg2)
        assertEquals("#ffffff", palettes[0].textColor)
    }

    @Test
    fun `decodeColorList respects a custom ab_range`() {
        val flat = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        val palettes = decodeColorList(flat, abRange = 0.3)
        // unit 0 always maps to L=0.5, a=0, b=0 regardless of ab_range;
        // gamma-corrected midpoint sRGB is 99 (0x63), not a naive 128 (0x80)
        assertEquals("#636363", palettes[0].bg1)
    }

    @Test
    fun `contrastRatio is 1 for identical colors and greater for black vs white`() {
        assertEquals(1.0, contrastRatio("#808080", "#808080"), 0.001)
        assertTrue(contrastRatio("#000000", "#ffffff") > 20.0)
    }

    @Test
    fun `fixContrast leaves already-sufficient contrast untouched`() {
        val palette = Palette(bg1 = "#ffffff", bg2 = "#ffffff", textColor = "#000000")
        val fixed = fixContrast(palette)
        assertEquals(palette, fixed)
    }

    @Test
    fun `argmax returns index of largest value`() {
        assertEquals(2, argmax(floatArrayOf(0.1f, 0.4f, 0.9f, 0.2f)))
    }

    @Test
    fun `softmax sums to 1`() {
        val out = softmax(floatArrayOf(1f, 2f, 3f))
        assertTrue(abs(out.sum() - 1.0f) < 0.0001f)
    }

    @Test
    fun `sigmoid maps 0 to 0_5`() {
        val out = sigmoid(floatArrayOf(0f))
        assertEquals(0.5, out[0].toDouble(), 0.0001)
    }
}
