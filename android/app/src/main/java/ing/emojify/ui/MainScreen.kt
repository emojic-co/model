package ing.emojify.ui

import android.content.Context
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalTextStyle
import androidx.compose.material3.MaterialTheme
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
import androidx.compose.runtime.withFrameNanos
import ing.emojify.cardGifFile
import ing.emojify.shareCardGif
import ing.emojify.model.NotoLottie
import ing.emojify.model.GifEncoder
import ing.emojify.ui.components.ExportState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.dp
import ing.emojify.model.EmojiCountPrefs
import ing.emojify.model.EmojiScore
import ing.emojify.model.Meta
import ing.emojify.ui.components.SWATCH_MAX_SIZE
import ing.emojify.ui.components.SWATCH_MIN_SIZE
import ing.emojify.model.OnnxPredictor
import ing.emojify.model.Palette
import ing.emojify.model.cycle
import ing.emojify.model.fixContrast
import ing.emojify.model.langForText
import ing.emojify.model.pickEmojiList
import ing.emojify.model.topFeelings
import ing.emojify.shareCard
import ing.emojify.ui.components.Card
import ing.emojify.ui.components.ColorBar
import ing.emojify.ui.components.EmojiList
import ing.emojify.ui.components.FeelingBar
import ing.emojify.ui.components.PatternBackground
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MIN_CHARS = 3
private const val DEBOUNCE_MS = 250L
private const val FEELING_COUNT = 5

private val DEFAULT_PALETTE = Palette(bg1 = "#a8e2f4", bg2 = "#78c9f4", textColor = "#282e36")

private val BACKGROUND_INK = Color(0xFF3A3A3A)
private val BACKGROUND_GRADIENT_COLORS = listOf(Color(0xFFFFE8D6), Color(0xFFFFD3E0), Color(0xFFAEE3F7))

private data class Override(val emoji: String? = null, val feeling: String? = null)

@Composable
fun MainScreen(meta: Meta, predictor: OnnxPredictor, onSettingsClick: () -> Unit) {
    var text by remember { mutableStateOf("") }
    var emojiTop by remember { mutableStateOf<List<EmojiScore>>(emptyList()) }
    var predictedFeeling by remember { mutableStateOf<String?>(null) }
    var feelingScores by remember { mutableStateOf<FloatArray?>(null) }
    var palettes by remember { mutableStateOf<List<Palette>>(emptyList()) }
    var override by remember { mutableStateOf(Override()) }
    var colorOverride by remember { mutableStateOf(0) }
    var lang by remember { mutableStateOf("en") }
    var capture by remember { mutableStateOf<(suspend () -> android.graphics.Bitmap)?>(null) }
    var exportPose by remember { mutableStateOf<ing.emojify.ui.components.ExportPose?>(null) }
    var exporting by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val emojiListState = rememberLazyListState()
    val feelingBarState = rememberLazyListState()
    val maxEmojis = remember {
        context.getSharedPreferences(EmojiCountPrefs.FILE, Context.MODE_PRIVATE)
            .getInt(EmojiCountPrefs.KEY_MAX_EMOJIS, EmojiCountPrefs.DEFAULT_MAX_EMOJIS)
            .coerceIn(EmojiCountPrefs.MIN_MAX_EMOJIS, EmojiCountPrefs.MAX_MAX_EMOJIS)
    }

    LaunchedEffect(text) {
        emojiListState.scrollToItem(0)
        feelingBarState.scrollToItem(0)
    }

    LaunchedEffect(text) {
        if (text.trim().length < MIN_CHARS) {
            emojiTop = emptyList()
            predictedFeeling = null
            feelingScores = null
            palettes = emptyList()
            override = Override()
            colorOverride = 0
            lang = "en"
            return@LaunchedEffect
        }
        delay(DEBOUNCE_MS)
        lang = langForText(text)
        val result = predictor.predict(text, meta)
        emojiTop = pickEmojiList(result.emojiLogits, meta.emojis, maxEmojis)
        val feelingIdx = ing.emojify.model.argmax(result.styleLogits)
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
                lang = lang,
                colors = colors,
                onShare = { scope.launch { capture?.invoke()?.let { shareCard(context, it) } } },
                onEmojiCycle = { dir -> override = override.copy(emoji = cycle(emojiTop.map { it.emoji }, shownEmoji, dir)) },
                onFeelingCycle = { dir -> override = override.copy(feeling = cycle(feelingOptions, shownFeeling, dir)) },
                onCaptureReady = { capture = it },
                export = exportPose,
                exportState = when {
                    exporting -> ExportState.Busy
                    shownEmoji != null && NotoLottie.assetPath(context, shownEmoji) != null -> ExportState.Ready
                    else -> ExportState.Disabled
                },
                onExportGif = {
                    val emoji = shownEmoji
                    val grab = capture
                    if (emoji != null && grab != null && !exporting) {
                        exporting = true
                        scope.launch {
                            try {
                                val cardStyle = ing.emojify.model.resolveFeeling(shownFeeling, lang)
                                val entranceMs = ing.emojify.model.entranceTotalMs(
                                    ing.emojify.model.Styles.file.textAnimations,
                                    cardStyle.entranceMotif,
                                    shownFeeling,
                                    ing.emojify.model.cardDisplayText(text, cardStyle),
                                ).toFloat()
                                exportCardGif(context, emoji, grab, entranceMs) { p -> exportPose = p }
                            } catch (e: Exception) {
                                android.util.Log.e("emojify", "gif export failed", e)
                                android.widget.Toast.makeText(context, "Export failed", android.widget.Toast.LENGTH_SHORT).show()
                            } finally {
                                exportPose = null
                                exporting = false
                            }
                        }
                    }
                },
            )
            Spacer(modifier = Modifier.height(16.dp))
            val clearButton: @Composable () -> Unit = {
                if (text.isNotEmpty()) {
                    IconButton(onClick = { text = "" }) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear text")
                    }
                }
            }
            val isHebrew = langForText(text) == "he"
            TextField(
                value = text,
                onValueChange = { text = it },
                placeholder = { Text("type at least 3 characters…") },
                textStyle = LocalTextStyle.current.copy(textDirection = TextDirection.Content),
                leadingIcon = if (isHebrew) clearButton else null,
                trailingIcon = if (isHebrew) null else clearButton,
                shape = RoundedCornerShape(16.dp),
                colors = TextFieldDefaults.colors(
                    unfocusedContainerColor = Color.White.copy(alpha = 0.6f),
                    focusedContainerColor = Color.White.copy(alpha = 0.8f),
                    unfocusedIndicatorColor = Color.Transparent,
                    focusedIndicatorColor = Color.Transparent,
                ),
                modifier = Modifier.fillMaxWidth(),
            )
            Text(
                "${text.length}/${meta.max_text_len}",
                style = MaterialTheme.typography.labelSmall,
                color = if (text.length >= meta.max_text_len) {
                    MaterialTheme.colorScheme.error
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                },
                textAlign = TextAlign.End,
                modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
            )
            Spacer(modifier = Modifier.height(16.dp))
            EmojiList(items = emojiTop.ifEmpty { null }, active = shownEmoji, state = emojiListState) { picked ->
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
                    FeelingBar(feelings = feelingOptions, active = shownFeeling, swatchSize = swatchSize, state = feelingBarState) { picked ->
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
        IconButton(
            onClick = onSettingsClick,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .windowInsetsPadding(WindowInsets.safeDrawing.exclude(WindowInsets.ime)),
        ) {
            Icon(Icons.Default.Settings, contentDescription = "Settings")
        }
    }
}

private const val EXPORT_FPS = 20
private const val GIF_SIZE_PX = 480

// Steps the card through the text entrance followed by one emoji loop (the shimmer plays one pass
// over that loop): poses each frame, waits for two composed frames, captures the card layer, then
// encodes a looping GIF and opens the share sheet.
private suspend fun exportCardGif(
    context: Context,
    emoji: String,
    capture: suspend () -> android.graphics.Bitmap,
    entranceMs: Float,
    setPose: (ing.emojify.ui.components.ExportPose) -> Unit,
) {
    val path = NotoLottie.assetPath(context, emoji) ?: return
    val durationMs = withContext(Dispatchers.IO) {
        com.airbnb.lottie.LottieCompositionFactory.fromAssetSync(context, path).value?.duration
    } ?: return
    val totalMs = entranceMs + durationMs
    val frames = kotlin.math.ceil(totalMs * EXPORT_FPS / 1000f).toInt().coerceAtLeast(1)
    val file = cardGifFile(context)
    val encoder = GifEncoder(file, GIF_SIZE_PX, delayCs = 100 / EXPORT_FPS)
    for (i in 0 until frames) {
        val t = i * 1000f / EXPORT_FPS
        setPose(
            ing.emojify.ui.components.ExportPose(
                lottie = ((t % durationMs) / durationMs).coerceIn(0f, 0.999f),
                textMs = t,
                shimmerPass = if (t >= entranceMs) ((t - entranceMs) / durationMs).coerceIn(0f, 1f) else null,
            ),
        )
        withFrameNanos { }
        withFrameNanos { }
        val bitmap = capture()
        withContext(Dispatchers.Default) { encoder.addFrame(bitmap) }
    }
    withContext(Dispatchers.Default) { encoder.finish() }
    shareCardGif(context, file)
}
