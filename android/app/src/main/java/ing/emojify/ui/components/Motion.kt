package ing.emojify.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.remember

@Composable
fun rememberEntranceScale(motif: String, durationMs: Int, key: Any?): State<Float> {
    val anim = remember(key) { Animatable(if (motif == "pop" || motif == "spin") 0.6f else 1f) }
    LaunchedEffect(key) { anim.animateTo(1f, tween(durationMs, easing = LinearOutSlowInEasing)) }
    return anim.asState()
}

@Composable
fun rememberEmojiBounce(motif: String, durationMs: Int): State<Float> {
    val transition = rememberInfiniteTransition(label = "emoji-$motif")
    val amplitude = when (motif) {
        "hop", "lift" -> 6f
        "shake", "tremor" -> 3f
        "breathe" -> 2f
        else -> 0f
    }
    return transition.animateFloat(
        initialValue = -amplitude,
        targetValue = amplitude,
        animationSpec = infiniteRepeatable(tween(durationMs, easing = LinearOutSlowInEasing), RepeatMode.Reverse),
        label = "emoji-offset",
    )
}
