package ing.emojify.model

import kotlinx.serialization.Serializable
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt

// Shared preview/export clip timeline. Rules live in web/src/clip.yml (ground truth, copied into
// style.yml as `clip`); semantics are in that file's header and mirrored by web/src/clip.js.

@Serializable
data class SpringSpec(val durationMs: Double, val from: Double, val zeta: Double, val omega: Double)

@Serializable
data class ClipSpec(
    val minEmojiLoops: Int,
    val maxClipMs: Double,
    val posterHoldMs: Int,
    val gifFps: Int,
    val mp4Fps: Int,
    val mp4EndHoldMs: Int = 0,
    val spring: SpringSpec,
)

data class ClipLength(val durationMs: Double, val loops: Int)

fun clipLength(entranceMs: Double, startDelayMs: Double, passMs: Double, loopMs: Double, spec: ClipSpec): ClipLength {
    val content = entranceMs + startDelayMs + passMs
    if (loopMs <= 0.0) return ClipLength(content, 0)
    val needed = maxOf(1, ceil(content / loopMs).toInt())
    var loops = maxOf(needed, spec.minEmojiLoops)
    while (loops > needed && loops * loopMs > spec.maxClipMs) loops -= 1
    return ClipLength(maxOf(content, minOf(loops * loopMs, spec.maxClipMs)), loops)
}

fun clipFrameCount(durationMs: Double, fps: Int): Int = maxOf(1, ceil(durationMs * fps / 1000.0).toInt())

fun springScale(tMs: Double, s: SpringSpec): Double {
    if (tMs >= s.durationMs) return 1.0
    val t = maxOf(0.0, tMs) / 1000.0
    val wd = s.omega * sqrt(1 - s.zeta * s.zeta)
    val decay = exp(-s.zeta * s.omega * t)
    val osc = cos(wd * t) + s.zeta * s.omega / wd * sin(wd * t)
    return 1 - (1 - s.from) * decay * osc
}

/** Pose sources for one card on the clip's time axis (t = ms since the clip started, looping at [durationMs]). */
class ClipTimeline(
    styles: StyleFile,
    motif: String,
    feeling: String?,
    cluster: String,
    unitCount: Int,
    val loopMs: Double,
) {
    val spec: ClipSpec = styles.clip!!
    val schedule: TextSchedule = textScheduleFor(styles.textAnimations!!, motif, feeling, unitCount)
    private val shimmerSpec = styles.shimmer!!
    private val effect = resolveShimmer(shimmerSpec, cluster, feeling)
    val shimmerStartMs: Double = schedule.totalMs + shimmerSpec.startDelayMs
    private val shimmerPlayer = ShimmerPlayer(effect, shimmerStartMs)
    private val length = clipLength(schedule.totalMs, shimmerSpec.startDelayMs, effect.durationMs, loopMs, spec)
    val durationMs: Double = length.durationMs
    val loops: Int = length.loops
    val shimmerEffect: ShimmerEffect get() = effect

    fun shimmerPoseAt(tMs: Double): ShimmerPose? = shimmerPlayer.poseAt(tMs)
}
