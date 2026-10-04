package ing.emojify.model

import kotlinx.serialization.Serializable

// Background shimmer effects. The numbers live in web/src/shimmers.yml (the ground truth, copied
// into style.yml as `shimmer`); the semantics are documented in that file's header and mirrored
// by web/src/shimmer.js. Keep the two in lockstep.

@Serializable
data class ShimmerStop(val at: Double, val color: String)

@Serializable
data class ShimmerKeyframe(val at: Double, val c: Double? = null, val opacity: Double? = null)

@Serializable
data class ShimmerEffect(
    val kind: String,
    val angleDeg: Double? = null,
    val widthFrac: Double? = null,
    val cx: Double? = null,
    val cy: Double? = null,
    val radiusFrac: Double? = null,
    val blend: String,
    val durationMs: Double,
    val easing: List<Double>,
    val stops: List<ShimmerStop>,
    val keyframes: List<ShimmerKeyframe>,
)

@Serializable
data class ShimmerOverride(
    val durationMs: Double? = null,
    val angleDeg: Double? = null,
)

@Serializable
data class ShimmerSpec(
    val startDelayMs: Double,
    val clusters: Map<String, ShimmerEffect>,
    val styles: Map<String, ShimmerOverride> = emptyMap(),
)

/** What to draw at one instant: the kind-specific number `c` and the layer opacity. */
data class ShimmerPose(val c: Float, val opacity: Float)

class ShimmerPlayer(val effect: ShimmerEffect, private val startMs: Double) {
    private val ease = cubicBezier(effect.easing[0], effect.easing[1], effect.easing[2], effect.easing[3])
    private val keyframes: List<Triple<Double, Double, Double>> = run {
        var c = effect.keyframes.first().c ?: 0.0
        var o = effect.keyframes.first().opacity ?: 1.0
        effect.keyframes.map { k ->
            c = k.c ?: c
            o = k.opacity ?: o
            Triple(k.at, c, o)
        }
    }

    /** Null while waiting to start, and once the single pass is over. */
    fun poseAt(elapsedMs: Double): ShimmerPose? {
        val t = elapsedMs - startMs
        if (t < 0.0) return null
        val u = t / effect.durationMs
        return if (u >= 1.0) null else poseAtPass(u)
    }

    /** The pose at pass progress [u] in 0..1 (used directly by the GIF export, one pass per loop). */
    fun poseAtPass(u: Double): ShimmerPose {
        var hi = keyframes.indexOfFirst { it.first >= u }
        if (hi < 0) hi = keyframes.lastIndex
        val a = if (hi > 0) keyframes[hi - 1] else keyframes[0]
        val b = keyframes[hi]
        val f = if (b.first == a.first) 1.0 else ease(((u - a.first) / (b.first - a.first)).coerceIn(0.0, 1.0))
        return ShimmerPose(
            c = (a.second + (b.second - a.second) * f).toFloat(),
            opacity = (a.third + (b.third - a.third) * f).toFloat(),
        )
    }
}

fun resolveShimmer(spec: ShimmerSpec, cluster: String, feeling: String?): ShimmerEffect {
    val base = spec.clusters[cluster] ?: spec.clusters.getValue("reflective")
    val o = spec.styles[feeling] ?: return base
    return base.copy(
        durationMs = o.durationMs ?: base.durationMs,
        angleDeg = o.angleDeg ?: base.angleDeg,
    )
}

/** "#RRGGBBAA" -> packed ARGB int. */
fun parseShimmerColor(hex: String): Int {
    val v = hex.removePrefix("#").toLong(16)
    val r = (v shr 24 and 0xFF).toInt()
    val g = (v shr 16 and 0xFF).toInt()
    val b = (v shr 8 and 0xFF).toInt()
    val a = (v and 0xFF).toInt()
    return (a shl 24) or (r shl 16) or (g shl 8) or b
}
