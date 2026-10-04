package ing.emojify.model

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ClipTest {
    private val file = parseStyleFile(File("src/main/assets/style.yml").readText())
    private val spec = file.clip!!

    @Test
    fun `clip ends when both emoji and shimmer are done`() {
        assertEquals(2750.0, clipLength(1000.0, 250.0, 1500.0, 1000.0, spec).durationMs, 0.0)
        assertEquals(4000.0, clipLength(1000.0, 250.0, 1500.0, 4000.0, spec).durationMs, 0.0)
        assertEquals(2750.0, clipLength(1000.0, 250.0, 1500.0, 0.0, spec).durationMs, 0.0)
    }

    @Test
    fun `frame count is ceil of length times fps`() {
        assertEquals(36, clipFrameCount(3000.0, spec.gifFps))
        assertEquals(60, clipFrameCount(3000.0, spec.mp4Fps))
        assertEquals(1, clipFrameCount(1.0, 12))
    }

    @Test
    fun `matches the web golden fixture`() {
        val cases = Json.parseToJsonElement(File("../../web/src/clipFixture.json").readText()).jsonArray
        for (c in cases) {
            val o = c.jsonObject
            val tl = ClipTimeline(
                file, o["motif"]!!.jsonPrimitive.content, o["feeling"]!!.jsonPrimitive.content,
                o["cluster"]!!.jsonPrimitive.content, o["unitCount"]!!.jsonPrimitive.int, o["loopMs"]!!.jsonPrimitive.double,
            )
            assertEquals(o["durationMs"]!!.jsonPrimitive.double, tl.durationMs, 1e-6)
            assertEquals(o["loops"]!!.jsonPrimitive.int, tl.loops)
            for (s in o["samples"]!!.jsonArray) {
                val so = s.jsonObject
                val t = so["t"]!!.jsonPrimitive.double
                val u0 = so["unit0"]!!.jsonObject
                val p0 = tl.schedule.poseAt(0, t)
                assertEquals(u0["opacity"]!!.jsonPrimitive.double, p0.opacity.toDouble(), 1e-3)
                assertEquals(u0["x"]!!.jsonPrimitive.double, p0.x.toDouble(), 1e-3)
                assertEquals(u0["scale"]!!.jsonPrimitive.double, p0.scale.toDouble(), 1e-3)
                val last = so["unitLast"]!!.jsonObject
                val pl = tl.schedule.poseAt(o["unitCount"]!!.jsonPrimitive.int - 1, t)
                assertEquals(last["opacity"]!!.jsonPrimitive.double, pl.opacity.toDouble(), 1e-3)
                val sh = so["shimmer"]
                val kp = tl.shimmerPoseAt(t)
                if (sh == null || sh is JsonNull) assertNull(kp)
                else {
                    assertEquals(sh.jsonObject["c"]!!.jsonPrimitive.double, kp!!.c.toDouble(), 1e-3)
                    assertEquals(sh.jsonObject["opacity"]!!.jsonPrimitive.double, kp.opacity.toDouble(), 1e-3)
                }
                val sp = so["spring"]
                if (sp != null && sp !is JsonNull) assertEquals(sp.jsonPrimitive.double, springScale(t, spec.spring), 1e-6)
            }
        }
    }
}
