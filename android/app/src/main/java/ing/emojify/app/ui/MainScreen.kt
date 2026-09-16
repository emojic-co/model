package ing.emojify.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import ing.emojify.app.copyCardToClipboard
import ing.emojify.app.model.EmojiScore
import ing.emojify.app.model.Meta
import ing.emojify.app.model.OnnxPredictor
import ing.emojify.app.model.Palette
import ing.emojify.app.model.fixContrast
import ing.emojify.app.model.pickEmojiList
import ing.emojify.app.model.topFeelings
import ing.emojify.app.shareCard
import ing.emojify.app.ui.components.Card
import ing.emojify.app.ui.components.ColorBar
import ing.emojify.app.ui.components.EmojiList
import ing.emojify.app.ui.components.FeelingBar
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MIN_CHARS = 3
private const val DEBOUNCE_MS = 250L
private const val EMOJI_SLOTS = 9
private const val FEELING_COUNT = 4

private val DEFAULT_PALETTE = Palette(bg1 = "#a8e2f4", bg2 = "#78c9f4", textColor = "#282e36")

private data class Override(val emoji: String? = null, val feeling: String? = null)

@Composable
fun MainScreen(meta: Meta, predictor: OnnxPredictor) {
    var text by remember { mutableStateOf("") }
    var emojiTop by remember { mutableStateOf<List<EmojiScore>>(emptyList()) }
    var predictedFeeling by remember { mutableStateOf<String?>(null) }
    var feelingScores by remember { mutableStateOf<FloatArray?>(null) }
    var palettes by remember { mutableStateOf<List<Palette>>(emptyList()) }
    var override by remember { mutableStateOf(Override()) }
    var colorOverride by remember { mutableStateOf(0) }
    var contrastFix by remember { mutableStateOf(true) }
    var capture by remember { mutableStateOf<(suspend () -> android.graphics.Bitmap)?>(null) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    LaunchedEffect(text) {
        if (text.trim().length < MIN_CHARS) {
            emojiTop = emptyList()
            predictedFeeling = null
            feelingScores = null
            palettes = emptyList()
            override = Override()
            colorOverride = 0
            return@LaunchedEffect
        }
        delay(DEBOUNCE_MS)
        val result = predictor.predict(text, meta)
        emojiTop = pickEmojiList(result.emojiLogits, meta.emojis, EMOJI_SLOTS)
        val feelingIdx = ing.emojify.app.model.argmax(result.styleLogits)
        predictedFeeling = meta.styles[feelingIdx]
        feelingScores = result.styleLogits
        palettes = result.palettes.ifEmpty { listOf(DEFAULT_PALETTE) }
        override = Override()
        colorOverride = 0
    }

    val shownEmoji = override.emoji ?: emojiTop.firstOrNull()?.emoji
    val shownFeeling = override.feeling ?: predictedFeeling
    val feelingOptions = topFeelings(feelingScores, meta.styles, shownFeeling, FEELING_COUNT)
    val displayPalettes = if (contrastFix) palettes.map { fixContrast(it) } else palettes
    val colors = displayPalettes.getOrElse(colorOverride) { DEFAULT_PALETTE }

    Column(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        TextField(
            value = text,
            onValueChange = { text = it },
            placeholder = { Text("type at least 3 characters…") },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(16.dp))
        EmojiList(items = emojiTop.ifEmpty { null }, active = shownEmoji) { picked ->
            override = override.copy(emoji = picked)
        }
        Spacer(modifier = Modifier.height(16.dp))
        Card(
            text = text,
            emoji = shownEmoji ?: "🙂",
            feeling = shownFeeling,
            colors = colors,
            onCopy = { scope.launch { capture?.invoke()?.let { copyCardToClipboard(context, it) } } },
            onShare = { scope.launch { capture?.invoke()?.let { shareCard(context, it) } } },
            onCaptureReady = { capture = it },
        )
        Spacer(modifier = Modifier.height(16.dp))
        ColorBar(palettes = displayPalettes, active = colorOverride) { colorOverride = it }
        Spacer(modifier = Modifier.height(16.dp))
        FeelingBar(feelings = feelingOptions, active = shownFeeling) { picked ->
            override = override.copy(feeling = picked)
        }
    }
}
