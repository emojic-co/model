package ing.emojify.ui.components

import android.provider.Settings
import androidx.compose.foundation.layout.Box
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.TextUnit
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.rememberLottieComposition
import ing.emojify.model.NotoLottie
import ing.emojify.model.SpringSpec
import ing.emojify.model.springScale

// The static glyph always takes up space (keeps layout, and is the fallback); once the Lottie
// clone has loaded it is drawn over the glyph, which turns transparent. [emojiMs] (draw phase only) is ms into
// the clip: Lottie loops with period [loopMs], the fallback spring (no clone) plays once; null = rest pose.
@Composable
fun AnimatedEmoji(
    emoji: String,
    fontSize: TextUnit,
    color: Color,
    spring: SpringSpec,
    loopMs: Float,
    emojiMs: () -> Float?,
    modifier: Modifier = Modifier,
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
    val springy = animate && path == null && !static
    Box(modifier = modifier) {
        Text(
            text = emoji,
            fontSize = fontSize,
            lineHeight = fontSize,
            style = TextStyle(platformStyle = PlatformTextStyle(includeFontPadding = false)),
            color = if (live) Color.Transparent else color,
            modifier = Modifier.graphicsLayer {
                val t = emojiMs()
                val s = if (springy && t != null) springScale(t.toDouble(), spring).toFloat() else 1f
                scaleX = s
                scaleY = s
            },
        )
        if (live) {
            val side = with(LocalDensity.current) { fontSize.toDp() }
            LottieAnimation(
                composition = composition,
                progress = {
                    val t = emojiMs()
                    if (t == null || loopMs <= 0f) 0f else ((t % loopMs) / loopMs).coerceIn(0f, 0.999f)
                },
                modifier = Modifier.align(Alignment.Center).size(side),
            )
        }
    }
}
