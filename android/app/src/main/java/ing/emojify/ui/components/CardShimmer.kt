package ing.emojify.ui.components

import android.provider.Settings
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import ing.emojify.model.ShimmerEffect
import ing.emojify.model.ShimmerPlayer
import ing.emojify.model.ShimmerSpec
import ing.emojify.model.parseShimmerColor
import ing.emojify.model.resolveShimmer
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

private fun blendModeFor(name: String) = when (name) {
    "screen" -> BlendMode.Screen
    "overlay" -> BlendMode.Overlay
    "softLight" -> BlendMode.Softlight
    "multiply" -> BlendMode.Multiply
    else -> BlendMode.SrcOver
}

/**
 * Background shimmer overlay (drawn above the card background/pattern, below emoji and text).
 * Starts [entranceMs] + spec.startDelayMs after [replayKey] changes, i.e. only once the text is
 * fully visible. Does nothing when [enabled] is false (share/GIF capture) or animations are off.
 */
@Composable
fun CardShimmer(
    spec: ShimmerSpec?,
    cluster: String,
    feeling: String?,
    entranceMs: Double,
    replayKey: Any?,
    enabled: Boolean,
    modifier: Modifier = Modifier,
) {
    if (spec == null || !enabled) return
    val context = LocalContext.current
    val animationsOff = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
    if (animationsOff) return
    val effect = remember(spec, cluster, feeling) { resolveShimmer(spec, cluster, feeling) }
    val player = remember(effect, entranceMs) { ShimmerPlayer(effect, entranceMs + spec.startDelayMs) }
    val stops = remember(effect) { effect.stops.map { it.at.toFloat() to Color(parseShimmerColor(it.color)) } }
    val blend = remember(effect) { blendModeFor(effect.blend) }
    var elapsedMs by remember(replayKey) { mutableLongStateOf(0L) }

    LaunchedEffect(replayKey) {
        val start = androidx.compose.runtime.withFrameNanos { it }
        while (true) androidx.compose.runtime.withFrameNanos { elapsedMs = (it - start) / 1_000_000 }
    }

    Canvas(modifier) {
        val pose = player.poseAt(elapsedMs.toDouble()) ?: return@Canvas
        val w = size.width
        val h = size.height
        val brush = shimmerBrush(effect, stops, pose.c, w, h) ?: return@Canvas
        drawRect(brush, alpha = pose.opacity.coerceIn(0f, 1f), blendMode = blend)
    }
}

// Geometry per the semantics in shimmers.yml: s(p) maps every card point to 0..1 along the axis.
private fun shimmerBrush(effect: ShimmerEffect, stops: List<Pair<Float, Color>>, c: Float, w: Float, h: Float): Brush? {
    val stopArray = stops.toTypedArray()
    val center = Offset(w / 2f, h / 2f)
    if (effect.kind == "radial") {
        val radius = c * (effect.radiusFrac ?: 0.5).toFloat() * w
        if (radius <= 0f) return null
        return Brush.radialGradient(*stopArray, center = Offset((effect.cx ?: 0.5).toFloat() * w, (effect.cy ?: 0.5).toFloat() * h), radius = radius)
    }
    val a = Math.toRadians(effect.angleDeg ?: 0.0)
    val dx = cos(a).toFloat()
    val dy = sin(a).toFloat()
    val extent = (abs(dx) + abs(dy)) * w // card extent along the axis, in px
    fun at(s: Float) = Offset(center.x + dx * (s - 0.5f) * extent, center.y + dy * (s - 0.5f) * extent)
    if (effect.kind == "flash") return Brush.linearGradient(*stopArray, start = at(0f), end = at(1f))
    val half = (effect.widthFrac ?: 0.2).toFloat() / 2f
    return Brush.linearGradient(*stopArray, start = at(c - half), end = at(c + half))
}
