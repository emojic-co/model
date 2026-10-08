package ing.emojify.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TextAnimationTest {
    private val anim: TextAnimations = parseStyleFile(File("src/main/assets/style.yml").readText()).textAnimations!!

    @Test
    fun `hash matches the web implementation`() {
        // Math.imul(i + 1, 2654435761) >>> 0 % 1000 / 1000
        assertEquals(0.761, textHash(0), 1e-9)
        assertEquals(textHash(3), textHash(3), 0.0)
        assertTrue(textHash(7) in 0.0..1.0)
    }

    @Test
    fun `splits words into graphemes`() {
        val units = splitTextUnits("hi שלום")
        assertEquals(listOf(2, 4), units.map { it.size })
        assertEquals(TextUnitRange(0, 1), units[0][0])
    }

    @Test
    fun `every entrance motif in style yml has an animation`() {
        val file = parseStyleFile(File("src/main/assets/style.yml").readText())
        file.styles.values.forEach { assertTrue(it.entrance in anim.motifs) }
        anim.styles.keys.forEach { assertTrue(it in file.styles) }
    }

    @Test
    fun `total duration is capped for long text`() {
        val s = textScheduleFor(anim, "rise", "Hopeful", 200)
        assertTrue(s.totalMs <= anim.timing.maxTotalMs + 1)
    }

    @Test
    fun `orders and overrides`() {
        assertEquals(0.0, textScheduleFor(anim, "drop", "Wistful", 3).delays[2], 0.0)
        val c = textScheduleFor(anim, "bloom", "Tender", 5).delays
        assertEquals(0.0, c[2], 0.0)
        assertEquals(c[0], c[4], 1e-9)
        assertEquals(140.0, textScheduleFor(anim, "settle", "Serene", 2).motif.staggerMs, 0.0)
    }

    @Test
    fun `pose starts hidden, ends at rest, and spin mirrors on odd units`() {
        val s = textScheduleFor(anim, "spin", "Playful", 4)
        assertEquals(0f, s.poseAt(0, 0.0).opacity, 0f)
        assertEquals(-140f, s.poseAt(0, -1.0).rotate, 1e-3f)
        assertEquals(140f, s.poseAt(1, -1.0).rotate, 1e-3f)
        val end = s.poseAt(2, s.totalMs)
        assertEquals(1f, end.opacity, 0f)
        assertEquals(0f, end.rotate, 0f)
        assertEquals(1f, end.scale, 0f)
    }
}
