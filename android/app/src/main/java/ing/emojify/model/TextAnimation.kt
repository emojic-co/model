package ing.emojify.model

import java.text.BreakIterator
import kotlinx.serialization.Serializable

// Per-character text entrance animations. The numbers live in web/src/textAnimations.yml (the
// ground truth, copied into style.yml as `textAnimations`); the semantics are documented in that
// file's header and mirrored by web/src/textAnimation.js. Keep the two in lockstep.

@Serializable
data class TextTiming(val maxTotalMs: Double)

@Serializable
data class TextKeyframe(
    val at: Double,
    val opacity: Double? = null,
    val x: Double? = null,
    val y: Double? = null,
    val scale: Double? = null,
    val scaleY: Double? = null,
    val rotate: Double? = null,
)

@Serializable
data class TextMotif(
    val durationMs: Double,
    val staggerMs: Double,
    val order: String,
    val jitterMs: Double,
    val alternate: Boolean,
    val easing: List<Double>,
    val keyframes: List<TextKeyframe>,
)

@Serializable
data class TextStyleOverride(
    val durationMs: Double? = null,
    val staggerMs: Double? = null,
    val order: String? = null,
    val jitterMs: Double? = null,
)

@Serializable
data class TextAnimations(
    val timing: TextTiming,
    val motifs: Map<String, TextMotif>,
    val styles: Map<String, TextStyleOverride> = emptyMap(),
)

/** A resolved keyframe: every field filled (carried forward from the previous keyframe). */
data class FullKeyframe(
    val at: Double,
    val opacity: Double,
    val x: Double,
    val y: Double,
    val scale: Double,
    val scaleY: Double,
    val rotate: Double,
)

data class UnitPose(
    val opacity: Float,
    val x: Float,
    val y: Float,
    val scale: Float,
    val scaleY: Float,
    val rotate: Float,
)

/** One animated unit: a grapheme (or a whole connected-script word) as a UTF-16 range. */
data class TextUnitRange(val start: Int, val end: Int)

class TextSchedule(
    val motif: TextMotif,
    val delays: DoubleArray,
    val keyframes: List<FullKeyframe>,
) {
    val totalMs: Double = (delays.maxOrNull() ?: 0.0) + motif.durationMs

    private val ease = cubicBezier(motif.easing[0], motif.easing[1], motif.easing[2], motif.easing[3])

    fun isDone(i: Int, elapsedMs: Double): Boolean = elapsedMs - delays[i] >= motif.durationMs

    fun poseAt(i: Int, elapsedMs: Double): UnitPose {
        val t = elapsedMs - delays[i]
        val p = when {
            t <= 0.0 -> 0.0
            t >= motif.durationMs -> 1.0
            else -> ease(t / motif.durationMs)
        }
        val flip = if (motif.alternate && i % 2 == 1) -1.0 else 1.0
        val ks = keyframes
        var hi = ks.indexOfFirst { it.at >= p }
        if (hi < 0) hi = ks.lastIndex
        val a = ks[maxOf(hi - 1, 0)].takeIf { hi > 0 } ?: ks[0]
        val b = ks[hi]
        val f = if (b.at == a.at) 1.0 else ((p - a.at) / (b.at - a.at)).coerceIn(0.0, 1.0)
        fun lerp(u: Double, v: Double) = u + (v - u) * f
        return UnitPose(
            opacity = lerp(a.opacity, b.opacity).toFloat(),
            x = (lerp(a.x, b.x) * flip).toFloat(),
            y = lerp(a.y, b.y).toFloat(),
            scale = lerp(a.scale, b.scale).toFloat(),
            scaleY = lerp(a.scaleY, b.scaleY).toFloat(),
            rotate = (lerp(a.rotate, b.rotate) * flip).toFloat(),
        )
    }
}

fun textHash(i: Int): Double = (((i + 1) * 2654435761L and 0xFFFFFFFFL) % 1000L) / 1000.0

private fun isConnectedScript(cp: Int): Boolean = when (Character.UnicodeScript.of(cp)) {
    Character.UnicodeScript.ARABIC, Character.UnicodeScript.DEVANAGARI, Character.UnicodeScript.THAI -> true
    else -> false
}

/** Words split on whitespace; graphemes within a word, except connected-script words stay whole. */
fun splitTextUnits(text: String): List<List<TextUnitRange>> {
    val words = ArrayList<List<TextUnitRange>>()
    var i = 0
    while (i < text.length) {
        while (i < text.length && Character.isWhitespace(text[i])) i++
        if (i >= text.length) break
        val start = i
        while (i < text.length && !Character.isWhitespace(text[i])) i++
        val word = text.substring(start, i)
        if (word.codePoints().anyMatch { isConnectedScript(it) }) {
            words.add(listOf(TextUnitRange(start, i)))
        } else {
            val bi = BreakIterator.getCharacterInstance()
            bi.setText(word)
            val units = ArrayList<TextUnitRange>()
            var s = bi.first()
            var e = bi.next()
            while (e != BreakIterator.DONE) {
                units.add(TextUnitRange(start + s, start + e))
                s = e
                e = bi.next()
            }
            words.add(units)
        }
    }
    return words
}

private fun fillKeyframes(m: TextMotif): List<FullKeyframe> {
    var cur = FullKeyframe(0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0)
    return m.keyframes.map { k ->
        cur = FullKeyframe(
            at = k.at,
            opacity = k.opacity ?: cur.opacity,
            x = k.x ?: cur.x,
            y = k.y ?: cur.y,
            scale = k.scale ?: cur.scale,
            scaleY = k.scaleY ?: cur.scaleY,
            rotate = k.rotate ?: cur.rotate,
        )
        cur
    }
}

fun resolveTextMotif(anim: TextAnimations, motif: String, feeling: String?): TextMotif {
    val base = anim.motifs[motif] ?: anim.motifs.getValue("settle")
    val o = anim.styles[feeling] ?: return base
    return base.copy(
        durationMs = o.durationMs ?: base.durationMs,
        staggerMs = o.staggerMs ?: base.staggerMs,
        order = o.order ?: base.order,
        jitterMs = o.jitterMs ?: base.jitterMs,
    )
}

fun textScheduleFor(anim: TextAnimations, motif: String, feeling: String?, n: Int): TextSchedule {
    val m = resolveTextMotif(anim, motif, feeling)
    val ranks = DoubleArray(n) { i ->
        when (m.order) {
            "reverse" -> (n - 1 - i).toDouble()
            "center" -> Math.abs(i - (n - 1) / 2.0)
            "random" -> textHash(i) * (n - 1)
            else -> i.toDouble()
        }
    }
    val maxRank = ranks.maxOrNull() ?: 0.0
    val cap = (anim.timing.maxTotalMs - m.durationMs - m.jitterMs) / (if (maxRank == 0.0) 1.0 else maxRank)
    val s = if (maxRank == 0.0) m.staggerMs else maxOf(0.0, minOf(m.staggerMs, cap))
    val delays = DoubleArray(n) { i -> ranks[i] * s + m.jitterMs * textHash(i) }
    return TextSchedule(m, delays, fillKeyframes(m))
}

/** CSS-style cubic-bezier(x1, y1, x2, y2) easing: progress in, eased progress out. */
fun cubicBezier(x1: Double, y1: Double, x2: Double, y2: Double): (Double) -> Double {
    fun coord(t: Double, a: Double, b: Double) = 3 * a * (1 - t) * (1 - t) * t + 3 * b * (1 - t) * t * t + t * t * t
    return { x ->
        var lo = 0.0
        var hi = 1.0
        repeat(40) {
            val mid = (lo + hi) / 2
            if (coord(mid, x1, x2) < x) lo = mid else hi = mid
        }
        coord((lo + hi) / 2, y1, y2)
    }
}
