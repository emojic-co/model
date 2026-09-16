package ing.emojify.app.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.exclude
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import ing.emojify.app.copyCardToClipboard
import ing.emojify.app.model.EmojiScore
import ing.emojify.app.model.Meta
import ing.emojify.app.ui.components.SWATCH_MAX_SIZE
import ing.emojify.app.ui.components.SWATCH_MIN_SIZE
import ing.emojify.app.model.OnnxPredictor
import ing.emojify.app.model.Palette
import ing.emojify.app.model.cycle
import ing.emojify.app.model.fixContrast
import ing.emojify.app.model.pickEmojiList
import ing.emojify.app.model.topFeelings
import ing.emojify.app.shareCard
import ing.emojify.app.ui.components.Card
import ing.emojify.app.ui.components.ColorBar
import ing.emojify.app.ui.components.EmojiList
import ing.emojify.app.ui.components.FeelingBar
import ing.emojify.app.ui.components.PatternBackground
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MIN_CHARS = 3
private const val DEBOUNCE_MS = 250L
private const val EMOJI_SLOTS = 10
private const val FEELING_COUNT = 5

private val DEFAULT_PALETTE = Palette(bg1 = "#a8e2f4", bg2 = "#78c9f4", textColor = "#282e36")

private val BACKGROUND_INK = Color(0xFF3A3A3A)
private val BACKGROUND_GRADIENT_COLORS = listOf(Color(0xFFFFE8D6), Color(0xFFFFD3E0), Color(0xFFAEE3F7))

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
    val displayPalettes = palettes.map { fixContrast(it) }
    val colors = displayPalettes.getOrElse(colorOverride) { DEFAULT_PALETTE }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .drawWithCache {
                val brush = Brush.linearGradient(
                    colors = BACKGROUND_GRADIENT_COLORS,
                    start = Offset(0f, 0f),
                    end = Offset(size.width, size.height),
                )
                onDrawBehind { drawRect(brush) }
            },
    ) {
        PatternBackground(
            cluster = "background",
            tint = BACKGROUND_INK,
            opacity = 0.05f,
            tilePx = 480,
            modifier = Modifier.fillMaxSize(),
        )
        Column(
            modifier = Modifier
                .fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing.exclude(WindowInsets.ime))
                .padding(16.dp),
        ) {
            Card(
                text = text,
                emoji = shownEmoji ?: "🙂",
                feeling = shownFeeling,
                colors = colors,
                onCopy = { scope.launch { capture?.invoke()?.let { copyCardToClipboard(context, it) } } },
                onShare = { scope.launch { capture?.invoke()?.let { shareCard(context, it) } } },
                onEmojiCycle = { dir -> override = override.copy(emoji = cycle(emojiTop.map { it.emoji }, shownEmoji, dir)) },
                onFeelingCycle = { dir -> override = override.copy(feeling = cycle(feelingOptions, shownFeeling, dir)) },
                onCaptureReady = { capture = it },
            )
            Spacer(modifier = Modifier.height(16.dp))
            TextField(
                value = text,
                onValueChange = { text = it },
                placeholder = { Text("type at least 3 characters…") },
                shape = RoundedCornerShape(16.dp),
                colors = TextFieldDefaults.colors(
                    unfocusedContainerColor = Color.White.copy(alpha = 0.6f),
                    focusedContainerColor = Color.White.copy(alpha = 0.8f),
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedIndicatorColor = Color.Transparent,
                ),
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(modifier = Modifier.height(16.dp))
            EmojiList(items = emojiTop.ifEmpty { null }, active = shownEmoji) { picked ->
                override = override.copy(emoji = picked)
            }
            Spacer(modifier = Modifier.height(16.dp))
            ColorBar(palettes = displayPalettes, active = colorOverride) { colorOverride = it }
            Spacer(modifier = Modifier.height(16.dp))
            Column(modifier = Modifier.weight(1f).fillMaxWidth()) {
                BoxWithConstraints(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentAlignment = Alignment.Center,
                ) {
                    val swatchSize = (maxHeight - 8.dp).coerceIn(SWATCH_MIN_SIZE, SWATCH_MAX_SIZE)
                    FeelingBar(feelings = feelingOptions, active = shownFeeling, swatchSize = swatchSize) { picked ->
                        override = override.copy(feeling = picked)
                    }
                }
                Text(
                    "made with ❤️ by Gilad",
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
    }
}
