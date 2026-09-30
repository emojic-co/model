package ing.emojify.ui.components

import android.provider.Settings
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.draw.scale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.TextUnit
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.rememberLottieComposition
import ing.emojify.model.NotoLottie

/** Length of the fallback spring, in ms; also the export progress scale (progress 1 = settled). */
const val SPRING_MS = 900f

/** Subtle under-damped spring from a slightly shrunk glyph to full size, [tMs] after it starts. */
fun springScale(tMs: Float): Float {
    if (tMs >= SPRING_MS) return 1f
    val t = tMs / 1000f
    val zeta = 0.4f
    val w = 14f
    val wd = w * kotlin.math.sqrt(1 - zeta * zeta)
    val decay = kotlin.math.exp(-zeta * w * t)
    val osc = kotlin.math.cos(wd * t) + zeta * w / wd * kotlin.math.sin(wd * t)
    return 1f - (1f - 0.88f) * decay * osc
}

// The static glyph always takes up space (keeps layout, and is the fallback); once the Lottie
// clone has loaded it is drawn over the glyph, which turns transparent. Preview only: the shared card image is captured with [static] set.
@Composable
fun AnimatedEmoji(
    emoji: String,
    fontSize: TextUnit,
    color: Color,
    modifier: Modifier = Modifier,
    progress: Float? = null,
    // Draw only the plain glyph (no Lottie frame, no spring scale); used when capturing the shared still image.
    static: Boolean = false,
) {
    val context = LocalContext.current
    val animate = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) > 0f
    }
    val path = if (animate) remember(emoji) { NotoLottie.assetPath(context, emoji) } else null
    val composition by rememberLottieComposition(
        if (path != null) LottieCompositionSpec.Asset(path) else LottieCompositionSpec.JsonString("{}"),
    )
    val live = !static && path != null && composition != null
    // No Lottie clone for this emoji: the glyph does a small spring scale instead.
    val spring = animate && path == null
    val springMs = remember { Animatable(SPRING_MS) }
    if (spring && progress == null) {
        LaunchedEffect(emoji) {
            springMs.snapTo(0f)
            springMs.animateTo(SPRING_MS, tween(SPRING_MS.toInt(), easing = LinearEasing))
        }
    }
    val scale = when {
        static || !spring -> 1f
        progress != null -> springScale(progress * SPRING_MS)
        else -> springScale(springMs.value)
    }
    Box(modifier = modifier) {
        Text(text = emoji, fontSize = fontSize, lineHeight = fontSize, style = TextStyle(platformStyle = PlatformTextStyle(includeFontPadding = false)), color = if (live) Color.Transparent else color, modifier = Modifier.scale(scale))
        if (live) {
            val side = with(LocalDensity.current) { fontSize.toDp() }
            val lottieModifier = Modifier.align(Alignment.Center).size(side)
            if (progress != null) {
                LottieAnimation(composition = composition, progress = { progress }, modifier = lottieModifier)
            } else {
                LottieAnimation(
                    composition = composition,
                    iterations = LottieConstants.IterateForever,
                    modifier = lottieModifier,
                )
            }
        }
    }
}
