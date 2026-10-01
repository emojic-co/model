package ing.emojify.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import ing.emojify.model.ClipTimeline
import ing.emojify.model.ShimmerEffect
import ing.emojify.model.parseShimmerColor
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
 * Background shimmer overlay (drawn above the card background/pattern, below emoji and text), posed from the
 * shared clip: [timeMs] (draw phase only) is ms into the clip, null = nothing drawn.
 */
@Composable
fun CardShimmer(
    timeline: ClipTimeline?,
    timeMs: () -> Float?,
    modifier: Modifier = Modifier,
) {
    if (timeline == null) return
    val effect = timeline.shimmerEffect
    val stops = remember(effect) { effect.stops.map { it.at.toFloat() to Color(parseShimmerColor(it.color)) } }
    val blend = remember(effect) { blendModeFor(effect.blend) }

    Canvas(modifier) {
        val t = timeMs() ?: return@Canvas
        val pose = timeline.shimmerPoseAt(t.toDouble()) ?: return@Canvas
        val brush = shimmerBrush(effect, stops, pose.c, size.width, size.height) ?: return@Canvas
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
