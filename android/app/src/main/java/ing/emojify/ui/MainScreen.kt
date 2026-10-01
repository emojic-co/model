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
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.ui.draw.alpha
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
import ing.emojify.model.ClipTimeline
import ing.emojify.model.Styles
import ing.emojify.model.cardDisplayText
import ing.emojify.model.clipFrameCount
import ing.emojify.model.loadEmojiLoopMs
import ing.emojify.model.splitTextUnits
import ing.emojify.ui.components.ShareFormat
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
import ing.emojify.model.ColorCountPrefs
import ing.emojify.model.DebouncePrefs
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
import ing.emojify.shareCardJpg
import ing.emojify.shareCardMp4
import ing.emojify.cardMp4File
import ing.emojify.model.Mp4Encoder
import ing.emojify.ui.components.Card
import ing.emojify.ui.components.ColorBar
import ing.emojify.ui.components.EmojiList
import ing.emojify.ui.components.FeelingBar
import ing.emojify.ui.components.PatternBackground
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val MIN_CHARS = 3
private const val FEELING_COUNT = 5

private val DEFAULT_PALETTE = Palette(bg1 = "#a8e2f4", bg2 = "#78c9f4", textColor = "#282e36")

private val BACKGROUND_INK = Color(0xFF3A3A3A)
private val BACKGROUND_GRADIENT_COLORS = listOf(Color(0xFFFFE8D6), Color(0xFFFFD3E0), Color(0xFFAEE3F7))

// The model emits a fixed handful of palettes per run and samples fresh noise each time,
// so more runs are appended until there are `count` of them.
private suspend fun collectPalettes(first: List<Palette>, count: Int, more: suspend () -> List<Palette>): List<Palette> {
    val all = first.toMutableList()
    while (all.isNotEmpty() && all.size < count) {
        val extra = more()
        if (extra.isEmpty()) break
        all += extra
    }
    return all.take(count)
}

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
    var exportPose by remember { mutableStateOf<ing.emojify.ui.components.ClipPose?>(null) }
    var exporting by remember { mutableStateOf<ShareFormat?>(null) }
    var exportProgress by remember { mutableFloatStateOf(0f) }
    var exportJob by remember { mutableStateOf<kotlinx.coroutines.Job?>(null) }
    val idle = exporting == null
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val emojiListState = rememberLazyListState()
    val feelingBarState = rememberLazyListState()
    val maxEmojis = remember {
        context.getSharedPreferences(EmojiCountPrefs.FILE, Context.MODE_PRIVATE)
            .getInt(EmojiCountPrefs.KEY_MAX_EMOJIS, EmojiCountPrefs.DEFAULT_MAX_EMOJIS)
            .coerceIn(EmojiCountPrefs.MIN_MAX_EMOJIS, EmojiCountPrefs.MAX_MAX_EMOJIS)
    }

    val colorCount = remember {
        context.getSharedPreferences(ColorCountPrefs.FILE, Context.MODE_PRIVATE)
            .getInt(ColorCountPrefs.KEY_COLOR_COUNT, ColorCountPrefs.DEFAULT_COLOR_COUNT)
            .coerceIn(ColorCountPrefs.MIN_COLOR_COUNT, ColorCountPrefs.MAX_COLOR_COUNT)
    }

    // The card and the prediction both wait for this much idle time after the last keystroke.
    val debounceMs = remember {
        context.getSharedPreferences(DebouncePrefs.FILE, Context.MODE_PRIVATE)
            .getInt(DebouncePrefs.KEY_DEBOUNCE_MS, DebouncePrefs.DEFAULT_DEBOUNCE_MS)
            .coerceIn(DebouncePrefs.MIN_DEBOUNCE_MS, DebouncePrefs.MAX_DEBOUNCE_MS)
            .toLong()
    }
    var cardText by remember { mutableStateOf("") }

    LaunchedEffect(text) {
        if (text.isEmpty()) {
            cardText = ""
        } else {
            delay(debounceMs)
            cardText = text
        }
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
        delay(debounceMs)
        lang = langForText(text)
        val result = predictor.predict(text, meta)
        emojiTop = pickEmojiList(result.emojiLogits, meta.emojis, maxEmojis)
        val feelingIdx = ing.emojify.model.argmax(result.styleLogits)
        predictedFeeling = meta.styles[feelingIdx]
        feelingScores = result.styleLogits
        palettes = collectPalettes(result.palettes, colorCount) {
            withContext(Dispatchers.Default) { predictor.predict(text, meta).palettes }
        }.ifEmpty { listOf(DEFAULT_PALETTE) }
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
                text = cardText,
                emoji = shownEmoji ?: "🙂",
                feeling = shownFeeling,
                lang = lang,
                colors = colors,
                onShare = { format ->
                    val emoji = shownEmoji
                    val grab = capture
                    if (emoji != null && grab != null && exporting == null) {
                        exporting = format
                        exportProgress = 0f
                        exportJob = scope.launch {
                            try {
                                if (format == ShareFormat.Jpg) {
                                    shareCardJpg(context, grab(), exportSizePx(context))
                                } else {
                                    val cardStyle = ing.emojify.model.resolveFeeling(shownFeeling, lang)
                                    val unitCount = splitTextUnits(cardDisplayText(cardText, cardStyle)).sumOf { it.size }
                                    val timeline = ClipTimeline(
                                        Styles.file, cardStyle.entranceMotif, shownFeeling, cardStyle.cluster, unitCount,
                                        loadEmojiLoopMs(context, emoji),
                                    )
                                    exportCardAnimation(context, format, grab, timeline, { exportProgress = it }) { p -> exportPose = p }
                                }
                            } catch (e: kotlinx.coroutines.CancellationException) {
                                throw e
                            } catch (e: Exception) {
                                android.util.Log.e("emojify", "${format.label} export failed", e)
                                android.widget.Toast.makeText(context, "Export failed", android.widget.Toast.LENGTH_SHORT).show()
                            } finally {
                                exportPose = null
                                exporting = null
                                exportJob = null
                            }
                        }
                    }
                },
                onCancelExport = { exportJob?.cancel() },
                exportProgress = exportProgress,
                onEmojiCycle = { dir -> if (idle) override = override.copy(emoji = cycle(emojiTop.map { it.emoji }, shownEmoji, dir)) },
                onFeelingCycle = { dir -> if (idle) override = override.copy(feeling = cycle(feelingOptions, shownFeeling, dir)) },
                onCaptureReady = { capture = it },
                pose = exportPose,
                busyFormat = exporting,
                shareEnabled = shownEmoji != null,
            )
            Spacer(modifier = Modifier.height(16.dp))
            val clearButton: @Composable () -> Unit = {
                if (text.isNotEmpty()) {
                    IconButton(onClick = { text = "" }, enabled = idle) {
                        Icon(Icons.Default.Clear, contentDescription = "Clear text")
                    }
                }
            }
            val isHebrew = langForText(text) == "he"
            TextField(
                value = text,
                onValueChange = { text = it },
                enabled = idle,
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
            Box(modifier = Modifier.alpha(if (idle) 1f else 0.4f)) {
                EmojiList(items = emojiTop.ifEmpty { null }, active = shownEmoji, state = emojiListState) { picked ->
                    if (idle) override = override.copy(emoji = picked)
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
            Box(modifier = Modifier.alpha(if (idle) 1f else 0.4f)) {
                ColorBar(palettes = displayPalettes, active = colorOverride) { if (idle) colorOverride = it }
            }
            Spacer(modifier = Modifier.height(16.dp))
            Column(modifier = Modifier.weight(1f).fillMaxWidth()) {
                BoxWithConstraints(
                    modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentAlignment = Alignment.Center,
                ) {
                    val swatchSize = (maxHeight - 8.dp).coerceIn(SWATCH_MIN_SIZE, SWATCH_MAX_SIZE)
                    Box(modifier = Modifier.alpha(if (idle) 1f else 0.4f)) {
                        FeelingBar(feelings = feelingOptions, active = shownFeeling, swatchSize = swatchSize, state = feelingBarState) { picked ->
                            if (idle) override = override.copy(feeling = picked)
                        }
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
            enabled = idle,
            modifier = Modifier
                .align(Alignment.BottomStart)
                .windowInsetsPadding(WindowInsets.safeDrawing.exclude(WindowInsets.ime)),
        ) {
            Icon(Icons.Default.Settings, contentDescription = "Settings")
        }
    }
}

private fun exportSizePx(context: Context): Int =
    context.getSharedPreferences(ing.emojify.model.ExportSizePrefs.FILE, Context.MODE_PRIVATE)
        .getInt(ing.emojify.model.ExportSizePrefs.KEY_SIZE_PX, ing.emojify.model.ExportSizePrefs.DEFAULT_SIZE_PX)

private const val GIF_SIZE_PX = 480

// Frame 0 is the poster (finished card, no shimmer, emoji at rest), held for ClipSpec.posterHoldMs so viewers
// that pause a GIF show a finished card. Then the shared clip (model/Clip.kt) is sampled at the format's fps.
// Each frame is posed, waits for two composed frames, and the card layer is captured; the frames are encoded
// into a looping GIF or MP4 and the share sheet opens.
private suspend fun exportCardAnimation(
    context: Context,
    format: ShareFormat,
    capture: suspend () -> android.graphics.Bitmap,
    timeline: ClipTimeline,
    onProgress: (Float) -> Unit,
    setPose: (ing.emojify.ui.components.ClipPose) -> Unit,
) {
    val spec = timeline.spec
    val fps = if (format == ShareFormat.Gif) spec.gifFps else spec.mp4Fps
    val frames = clipFrameCount(timeline.durationMs, fps)
    val gifFile = cardGifFile(context)
    val mp4File = cardMp4File(context)
    // GIF delays are whole centiseconds (8 cs at 12 fps).
    val gif = if (format == ShareFormat.Gif) GifEncoder(gifFile, GIF_SIZE_PX, delayCs = 100 / fps) else null
    val mp4 = if (format == ShareFormat.Mp4) Mp4Encoder(mp4File, exportSizePx(context), fps) else null

    suspend fun grab(pose: ing.emojify.ui.components.ClipPose, delayMs: Int) {
        setPose(pose)
        withFrameNanos { }
        withFrameNanos { }
        val bitmap = capture()
        withContext(Dispatchers.Default) {
            gif?.addFrame(bitmap, delayMs / 10)
            mp4?.addFrame(bitmap, delayMs)
        }
    }

    grab(ing.emojify.ui.components.ClipPose(0f, poster = true), spec.posterHoldMs)
    for (i in 0 until frames) {
        onProgress(i / frames.toFloat())
        grab(ing.emojify.ui.components.ClipPose(i * 1000f / fps), 1000 / fps)
    }
    withContext(Dispatchers.Default) {
        gif?.finish()
        mp4?.finish()
    }
    if (format == ShareFormat.Gif) shareCardGif(context, gifFile) else shareCardMp4(context, mp4File)
}
