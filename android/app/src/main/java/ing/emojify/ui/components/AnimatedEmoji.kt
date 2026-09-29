package ing.emojify.ui.components

import android.provider.Settings
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.TextUnit
import com.airbnb.lottie.compose.LottieAnimation
import com.airbnb.lottie.compose.LottieCompositionSpec
import com.airbnb.lottie.compose.LottieConstants
import com.airbnb.lottie.compose.rememberLottieComposition
import ing.emojify.model.NotoLottie

// The static glyph always takes up space (keeps layout, and is the fallback); once the Lottie
// clone has loaded it is drawn over the glyph, which turns transparent. Preview only: the shared card image is captured as a static frame.
@Composable
fun AnimatedEmoji(
    emoji: String,
    fontSize: TextUnit,
    color: Color,
    modifier: Modifier = Modifier,
    progress: Float? = null,
) {
    val context = LocalContext.current
    val animate = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) > 0f
    }
    val path = if (animate) remember(emoji) { NotoLottie.assetPath(context, emoji) } else null
    val composition by rememberLottieComposition(
        if (path != null) LottieCompositionSpec.Asset(path) else LottieCompositionSpec.JsonString("{}"),
    )
    val live = path != null && composition != null
    Box(modifier = modifier) {
        Text(text = emoji, fontSize = fontSize, color = if (live) Color.Transparent else color)
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
