package ing.emojify.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ShimmerTest {
    private val file = parseStyleFile(File("src/main/assets/style.yml").readText())
    private val spec = file.shimmer!!

    @Test
    fun `every cluster used by a style has an effect and overrides name real styles`() {
        file.styles.values.forEach { assertTrue(it.cluster in spec.clusters) }
        spec.styles.keys.forEach { assertTrue(it in file.styles) }
    }

    @Test
    fun `overrides apply on top of the cluster effect`() {
        assertEquals(480.0, resolveShimmer(spec, "anger", "Furious").durationMs, 0.0)
        assertEquals(650.0, resolveShimmer(spec, "anger", "Irritated").durationMs, 0.0)
    }

    @Test
    fun `starts after the entrance plus start delay and pauses between passes`() {
        val e = resolveShimmer(spec, "drive", "Hopeful")
        val p = ShimmerPlayer(e, 1000.0 + spec.startDelayMs)
        assertNull(p.poseAt(0.0))
        assertNull(p.poseAt(1000.0 + spec.startDelayMs - 1))
        val first = p.poseAt(1000.0 + spec.startDelayMs)
        assertNotNull(first)
        assertEquals(-0.2f, first!!.c, 1e-4f)
        assertNull(p.poseAt(1000.0 + spec.startDelayMs + e.durationMs + 10))
        // second cycle starts again
        assertNotNull(p.poseAt(1000.0 + spec.startDelayMs + e.durationMs + e.pauseMs + 1))
    }

    @Test
    fun `sweep ends past the far edge and colors parse as RRGGBBAA`() {
        val e = resolveShimmer(spec, "drive", "Hopeful")
        val p = ShimmerPlayer(e, 0.0)
        assertEquals(1.2f, p.poseAt(e.durationMs - 0.001)!!.c, 1e-2f)
        assertEquals(0xCCFFFFFF.toInt(), parseShimmerColor("#FFFFFFCC"))
        assertEquals(0x00FF3B00, parseShimmerColor("#FF3B0000"))
    }
}
