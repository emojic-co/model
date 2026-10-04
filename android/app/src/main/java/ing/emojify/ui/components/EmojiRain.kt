package ing.emojify.ui.components

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.random.Random

const val EGG_PHRASE = "emojify.ing"
private val EGG_EMOJIS = listOf("🎉", "✨", "😀", "🥳", "🌈", "💖", "🚀", "🎈")
private const val EGG_COUNT = 40

fun isEasterEgg(text: String): Boolean = text.trim().equals(EGG_PHRASE, ignoreCase = true)

private data class Piece(val emoji: String, val x: Float, val delayMs: Int, val durationMs: Int, val sizeSp: Int)

// Easter egg: a rain of emojis across the whole screen. Re-keyed on `trigger` so typing the phrase again replays it.
@Composable
fun EmojiRain(trigger: Int, modifier: Modifier = Modifier) {
    if (trigger == 0) return
    val pieces = remember(trigger) {
        List(EGG_COUNT) {
            Piece(
                emoji = EGG_EMOJIS.random(),
                x = Random.nextFloat(),
                delayMs = Random.nextInt(1200),
                durationMs = 2000 + Random.nextInt(1500),
                sizeSp = 24 + Random.nextInt(24),
            )
        }
    }
    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val width = maxWidth
        val height = maxHeight
        pieces.forEach { p ->
            val progress = remember(trigger, p) { Animatable(0f) }
            LaunchedEffect(trigger, p) {
                progress.animateTo(1f, tween(p.durationMs, p.delayMs, LinearEasing))
            }
            if (progress.value in 0.0001f..0.9999f) {
                Text(
                    p.emoji,
                    fontSize = p.sizeSp.sp,
                    modifier = Modifier
                        .offset(x = width * p.x, y = (height + 48.dp) * progress.value - 48.dp)
                        .rotate(360f * progress.value),
                )
            }
        }
    }
}
