package ing.emojify.ui.components

import android.provider.Settings
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.ui.graphics.luminance
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.foundation.Image
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.layout.ContentScale
import kotlinx.coroutines.launch
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.PlatformTextStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.LineHeightStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ing.emojify.R
import ing.emojify.model.LocalStrings
import ing.emojify.model.Palette
import ing.emojify.model.Styles
import ing.emojify.model.cardDisplayText
import ing.emojify.model.ClipTimeline
import ing.emojify.model.loadEmojiLoopMs
import ing.emojify.model.splitTextUnits
import ing.emojify.model.patternTint
import ing.emojify.model.resolveFeeling

private val fontProvider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

// Fonts the Google provider doesn't serve ship in assets/fonts/<Family_Name>.ttf and win over it.
private fun cardFontFamily(context: android.content.Context, name: String, weight: Int, italic: Boolean): FontFamily {
    val file = "fonts/${name.replace(' ', '_')}.ttf"
    val bundled = try {
        context.assets.open(file).close()
        true
    } catch (_: java.io.IOException) {
        false
    }
    val fontStyle = if (italic) FontStyle.Italic else FontStyle.Normal
    return if (bundled) FontFamily(androidx.compose.ui.text.font.Font(file, context.assets, FontWeight(weight), fontStyle))
    else FontFamily(
        Font(
            googleFont = GoogleFont(name),
            fontProvider = fontProvider,
            weight = FontWeight(weight),
            style = fontStyle,
        ),
    )
}

private val watermarkFont = FontFamily(Font(googleFont = GoogleFont("Caveat"), fontProvider = fontProvider))

@Composable
fun Card(
    text: String,
    emoji: String,
    feeling: String?,
    lang: String?,
    colors: Palette,
    onShare: (ShareFormat) -> Unit,
    onEmojiCycle: (Int) -> Unit,
    onFeelingCycle: (Int) -> Unit,
    onCaptureReady: ((suspend () -> android.graphics.Bitmap) -> Unit)? = null,
    // Frame-stepped GIF/MP4 export: when set, the card is pinned to this clip pose instead of looping live.
    pose: ClipPose? = null,
    // Null while idle; the format being exported while busy (all share buttons are disabled meanwhile).
    busyFormat: ShareFormat? = null,
    shareEnabled: Boolean = true,
    // 0..1 while exporting; the share buttons give way to a progress bar and a cancel button.
    exportProgress: Float = 0f,
    onCancelExport: () -> Unit = {},
) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    val style = resolveFeeling(feeling, lang)
    val tint = Color(android.graphics.Color.parseColor(patternTint(colors.bg1, colors.bg2)))
    val context = LocalContext.current
    val fontFamily = remember(style.fontName, style.faceWeight, style.faceItalic) { cardFontFamily(context, style.fontName, style.faceWeight, style.faceItalic) }
    val displayText = cardDisplayText(text, style)
    val graphicsLayer = rememberGraphicsLayer()
    // Export steps the live card through the clip from the poster on, which would visibly reset the preview;
    // instead the frame on screen at the tap is frozen on top (outside the captured layer) until the export ends.
    var frozen by remember { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }
    val uiScope = rememberCoroutineScope()
    LaunchedEffect(busyFormat) { if (busyFormat == null) frozen = null }
    // Shared still image capture: finished card, no shimmer, plain emoji glyph.
    var stillCapture by remember { mutableStateOf(false) }
    val exporting by rememberUpdatedState(pose != null)
    val clipSpec = Styles.file.clip!!
    val animationsOff = remember {
        Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
    // The preview loops the same clip the exports sample (see model/Clip.kt); it restarts from t=0 on any card change.
    var loopMs by remember(emoji) { mutableStateOf<Double?>(null) }
    LaunchedEffect(emoji) { loopMs = loadEmojiLoopMs(context, emoji) }
    val unitCount = remember(displayText) { splitTextUnits(displayText).sumOf { it.size } }
    val timeline = remember(loopMs, style.entranceMotif, feeling, style.cluster, unitCount) {
        loopMs?.let { ClipTimeline(Styles.file, style.entranceMotif, feeling, style.cluster, unitCount, it) }
    }
    var previewMs by remember { mutableFloatStateOf(0f) }
    val restartKey = listOf(emoji, feeling, lang, displayText, colors)
    LaunchedEffect(timeline, restartKey, animationsOff, pose == null) {
        val tl = timeline ?: return@LaunchedEffect
        if (animationsOff || pose != null) return@LaunchedEffect
        previewMs = 0f
        val start = androidx.compose.runtime.withFrameNanos { it }
        while (true) androidx.compose.runtime.withFrameNanos { previewMs = ((it - start) / 1_000_000f) % tl.durationMs.toFloat() }
    }
    // Poster = finished card text, no shimmer, emoji at rest. Shown until the clip is ready, with animations off, and for stills.
    val poster = pose?.poster == true || stillCapture || timeline == null || (pose == null && animationsOff)
    val restEmojiMs = if ((loopMs ?: 0.0) > 0.0) 0f else clipSpec.spring.durationMs.toFloat()
    val clipMs: () -> Float = { pose?.tMs ?: previewMs }
    val textMsFn: () -> Float = { if (poster) TEXT_ANIM_DONE else clipMs() }
    val shimmerMsFn: () -> Float? = { if (poster) null else clipMs() }
    val emojiMsFn: () -> Float? = { if (poster) restEmojiMs else clipMs() }
    LaunchedEffect(onCaptureReady) {
        onCaptureReady?.invoke {
            if (exporting) {
                // Clip frame: the caller already posed the card for this frame.
                graphicsLayer.toImageBitmap().asAndroidBitmap()
            } else {
                // Shared images stay static: finished card, no shimmer, plain emoji glyph; let a frame draw.
                stillCapture = true
                androidx.compose.runtime.withFrameNanos { }
                androidx.compose.runtime.withFrameNanos { }
                try {
                    graphicsLayer.toImageBitmap().asAndroidBitmap()
                } finally {
                    stillCapture = false
                }
            }
        }
    }

    val global = Styles.file.global
    val shareInk = ShareInk
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val cardWidthDp = maxWidth
        val emojiSizeSp = cardWidthDp.value * global.emojiRatio
        val cardPadding = cardWidthDp * global.padRatio
        val cardPaddingTop = cardWidthDp * global.padTopRatio
        val cardPaddingBottom = cardWidthDp * global.padBottomRatio
        val emojiTextGap = cardWidthDp * global.emojiTextGapRatio
        val textMinSp = cardWidthDp.value * global.textMinRatio
        val textMaxSp = cardWidthDp.value * global.textMaxRatio
        val density = LocalDensity.current
        val patternTileWidthPx = with(density) { (cardWidthDp * style.patternWidthRatio).toPx() }.toInt().coerceAtLeast(1)
        val patternTileHeightPx = with(density) { (cardWidthDp * style.patternHeightRatio).toPx() }.toInt().coerceAtLeast(1)

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(RoundedCornerShape(16.dp))
                    .drawWithContent {
                        graphicsLayer.record { this@drawWithContent.drawContent() }
                        drawLayer(graphicsLayer)
                    }
                    .pointerInput(onEmojiCycle, onFeelingCycle) {
                        detectDragGestures(
                            onDragEnd = {},
                            onDrag = { change, dragAmount ->
                                change.consume()
                                if (kotlin.math.abs(dragAmount.x) > kotlin.math.abs(dragAmount.y)) {
                                    if (kotlin.math.abs(dragAmount.x) > 24) onEmojiCycle(if (dragAmount.x > 0) -1 else 1)
                                } else {
                                    if (kotlin.math.abs(dragAmount.y) > 24) onFeelingCycle(if (dragAmount.y > 0) -1 else 1)
                                }
                            },
                        )
                    },
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Brush.linearGradient(listOf(bg1, bg2))),
                )
                FeelingPatternBackground(
                    feeling = feeling ?: "Neutral",
                    svg = style.patternSvg,
                    tint = tint,
                    modifier = Modifier.matchParentSize(),
                    opacity = global.maxPatternOpacity,
                    tileWidthPx = patternTileWidthPx,
                    tileHeightPx = patternTileHeightPx,
                )
                CardShimmer(
                    timeline = timeline,
                    timeMs = shimmerMsFn,
                    modifier = Modifier.matchParentSize(),
                )
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(start = cardPadding, end = cardPadding, top = cardPaddingTop, bottom = cardPaddingBottom),
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        AnimatedEmoji(
                            emoji = emoji,
                            fontSize = emojiSizeSp.sp,
                            color = textColor,
                            spring = clipSpec.spring,
                            loopMs = (loopMs ?: 0.0).toFloat(),
                            emojiMs = emojiMsFn,
                            static = stillCapture,
                        )
                        Box(
                            modifier = Modifier.padding(top = emojiTextGap).weight(1f, fill = false).fillMaxWidth(),
                            contentAlignment = Alignment.Center,
                        ) {
                            BoxWithConstraints(
                                modifier = Modifier.fillMaxWidth(),
                            ) {
                                val density = LocalDensity.current
                                val maxWidthPx = with(density) { maxWidth.toPx() }.toInt()
                                val maxHeightPx = with(density) { maxHeight.toPx() }.toInt()
                                val fitSp = rememberFitFontSizeSp(
                                    text = displayText,
                                    fontFamily = fontFamily,
                                    fontWeight = FontWeight(style.fontWeight),
                                    fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                    letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    maxWidthPx = maxWidthPx,
                                    maxHeightPx = maxHeightPx,
                                    minSp = textMinSp,
                                    maxSp = textMaxSp,
                                    lineHeightMultiplier = global.textLineHeight,
                                )
                                AnimatedCharText(
                                    text = displayText,
                                    textStyle = TextStyle(
                                        textDirection = TextDirection.Content,
                                        textAlign = TextAlign.Center,
                                        fontFamily = fontFamily,
                                        fontSize = fitSp.sp,
                                        lineHeight = (fitSp * global.textLineHeight).sp,
                                        platformStyle = PlatformTextStyle(includeFontPadding = false),
                                        lineHeightStyle = LineHeightStyle(LineHeightStyle.Alignment.Center, LineHeightStyle.Trim.None),
                                        fontWeight = FontWeight(style.fontWeight),
                                        fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                        letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    ),
                                    color = textColor,
                                    maxWidthPx = maxWidthPx,
                                    animations = Styles.file.textAnimations,
                                    motif = style.entranceMotif,
                                    feeling = feeling,
                                    textMs = textMsFn,
                                    maxLines = global.maxLines,
                                )
                            }
                        }
                    }
                }
                // Watermark lives inside the captured layer so jpg/gif/mp4 all carry it.
                Text(
                    text = "emojify.ing",
                    color = (if (bg2.luminance() > 0.5f) Color.Black else Color.White).copy(alpha = global.watermarkOpacity),
                    fontFamily = watermarkFont,
                    fontWeight = FontWeight.Bold,
                    fontSize = (cardWidthDp.value * global.watermarkPxRatio).sp,
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(end = cardWidthDp * global.watermarkMarginRatio, top = cardWidthDp * global.watermarkMarginRatio),
                )
            }
            frozen?.let {
                Image(
                    bitmap = it,
                    contentDescription = null,
                    contentScale = ContentScale.FillBounds,
                    modifier = Modifier.fillMaxWidth().aspectRatio(1f).clip(RoundedCornerShape(16.dp)),
                )
            }
        }
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.padding(top = 12.dp),
            ) {
                if (busyFormat == null) {
                    for (format in ShareFormat.entries) {
                        ShareButton(
                            label = format.label,
                            enabled = shareEnabled,
                            busy = false,
                            onClick = {
                                uiScope.launch {
                                    if (format != ShareFormat.Jpg) frozen = runCatching { graphicsLayer.toImageBitmap() }.getOrNull()
                                    onShare(format)
                                }
                            },
                            ink = shareInk,
                        )
                    }
                } else {
                    LinearProgressIndicator(
                        progress = { exportProgress.coerceIn(0f, 1f) },
                        color = shareInk,
                        trackColor = shareInk.copy(alpha = 0.25f),
                        modifier = Modifier.align(Alignment.CenterVertically).width(140.dp),
                    )
                    ShareButton(label = LocalStrings.current.t("card.cancel"), enabled = true, busy = false, onClick = onCancelExport, ink = shareInk)
                }
            }
        }
    }
}

/** One export frame: ms into the shared clip, or the poster (finished card, no shimmer, emoji at rest). */
data class ClipPose(val tMs: Float, val poster: Boolean = false)

enum class ShareFormat(val label: String) { Jpg("jpg"), Gif("gif"), Mp4("mp4") }

@Composable
private fun ShareButton(
    label: String,
    onClick: () -> Unit,
    enabled: Boolean,
    busy: Boolean,
    ink: Color,
) {
    // Mirrors the web .share-bar button: solid ink pill, white bold uppercase spaced label, soft shadow.
    val shape = RoundedCornerShape(percent = 50)
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .alpha(if (enabled || busy) 1f else 0.35f)
            .shadow(if (enabled) 3.dp else 0.dp, shape)
            .clip(shape)
            .background(ink)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 20.dp, vertical = 9.dp),
    ) {
        if (busy) {
            CircularProgressIndicator(
                modifier = Modifier.size(14.dp),
                strokeWidth = 2.dp,
                color = Color.White,
            )
        } else {
            Text(label.uppercase(), color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 1.8.sp)
        }
    }
}

/** Share button / progress ink, same as the web --ink. */
private val ShareInk = Color(0xFF1A1A1A)
