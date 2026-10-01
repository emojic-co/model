package ing.emojify.ui.components

import androidx.compose.animation.core.Animatable
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.ui.graphics.luminance
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
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
import ing.emojify.model.Palette
import ing.emojify.model.Styles
import ing.emojify.model.cardDisplayText
import ing.emojify.model.entranceTotalMs
import ing.emojify.model.patternTint
import ing.emojify.model.resolveFeeling

private val fontProvider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

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
    // Frame-stepped GIF export: when set, emoji, text entrance and shimmer are pinned to this pose.
    export: ExportPose? = null,
    // Null while idle; the format being exported while busy (all share buttons are disabled meanwhile).
    busyFormat: ShareFormat? = null,
    shareEnabled: Boolean = true,
) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    val style = resolveFeeling(feeling, lang)
    val tint = Color(android.graphics.Color.parseColor(patternTint(colors.bg1, colors.bg2)))
    val fontFamily = FontFamily(Font(googleFont = GoogleFont(style.fontName), fontProvider = fontProvider))
    val displayText = cardDisplayText(text, style)
    val graphicsLayer = rememberGraphicsLayer()
    val textClock = remember { Animatable(0f) }
    var shimmerOn by remember { mutableStateOf(true) }
    var staticEmoji by remember { mutableStateOf(false) }
    val exporting by rememberUpdatedState(export != null)
    LaunchedEffect(onCaptureReady) {
        onCaptureReady?.invoke {
            if (exporting) {
                // GIF frame: the caller already posed the emoji, text and shimmer for this frame.
                graphicsLayer.toImageBitmap().asAndroidBitmap()
            } else {
                // Shared images stay static: finish the text entrance, hide the shimmer, show the plain emoji glyph, let a frame draw.
                textClock.snapTo(TEXT_ANIM_DONE)
                shimmerOn = false
                staticEmoji = true
                androidx.compose.runtime.withFrameNanos { }
                androidx.compose.runtime.withFrameNanos { }
                try {
                    graphicsLayer.toImageBitmap().asAndroidBitmap()
                } finally {
                    shimmerOn = true
                    staticEmoji = false
                }
            }
        }
    }

    val global = Styles.file.global
    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val cardWidthDp = maxWidth
        val emojiSizeSp = cardWidthDp.value * global.emojiRatio
        val cardPadding = cardWidthDp * global.padRatio
        val textMinSp = cardWidthDp.value * global.textMinRatio
        val textMaxSp = cardWidthDp.value * global.textMaxRatio
        val density = LocalDensity.current
        val patternTileWidthPx = with(density) { (cardWidthDp * style.patternWidthRatio).toPx() }.toInt().coerceAtLeast(1)
        val patternTileHeightPx = with(density) { (cardWidthDp * style.patternHeightRatio).toPx() }.toInt().coerceAtLeast(1)

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
                    spec = Styles.file.shimmer,
                    cluster = style.cluster,
                    feeling = feeling,
                    entranceMs = entranceTotalMs(Styles.file.textAnimations, style.entranceMotif, feeling, displayText),
                    replayKey = Triple(feeling, emoji, lang),
                    enabled = shimmerOn && (export == null || export.shimmerPass != null),
                    exportProgress = export?.shimmerPass,
                    modifier = Modifier.matchParentSize(),
                )
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(start = cardPadding, top = cardPadding, end = cardPadding, bottom = maxOf(cardPadding, SHARE_BAR_CLEARANCE)),
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center,
                    ) {
                        AnimatedEmoji(
                            emoji = emoji,
                            fontSize = emojiSizeSp.sp,
                            color = textColor,
                            progress = export?.lottie,
                            static = staticEmoji,
                            modifier = Modifier.offset(y = cardWidthDp * global.emojiDyRatio),
                        )
                        Spacer(Modifier.height(cardWidthDp * global.gapRatio))
                        Box(
                            modifier = Modifier.weight(1f, fill = false).fillMaxWidth(),
                            contentAlignment = Alignment.Center,
                        ) {
                            BoxWithConstraints(
                                modifier = Modifier.fillMaxWidth().padding(
                                    horizontal = cardWidthDp * global.textBoxPadXRatio,
                                    vertical = cardWidthDp * global.textBoxPadYRatio,
                                ),
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
                                        fontWeight = FontWeight(style.fontWeight),
                                        fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                        letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    ),
                                    color = textColor,
                                    maxWidthPx = maxWidthPx,
                                    animations = Styles.file.textAnimations,
                                    motif = style.entranceMotif,
                                    feeling = feeling,
                                    replayKey = Triple(feeling, emoji, lang),
                                    clock = textClock,
                                    exportMs = export?.textMs,
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
                        .align(Alignment.BottomEnd)
                        .padding(end = cardWidthDp * global.watermarkMarginRatio, bottom = cardWidthDp * global.watermarkMarginRatio),
                )
            }
            Row(
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                modifier = Modifier.align(Alignment.BottomCenter).padding(14.dp),
            ) {
                for (format in ShareFormat.entries) {
                    ShareButton(
                        label = format.label,
                        enabled = shareEnabled && busyFormat == null,
                        busy = busyFormat == format,
                        ink = textColor,
                        onClick = { onShare(format) },
                    )
                }
            }
        }
    }
}

/** Space reserved at the card bottom so text never sits under the share buttons (14dp inset + ~30dp button + breathing room). */
private val SHARE_BAR_CLEARANCE = 58.dp

/** One GIF frame: emoji loop progress, ms into the text entrance, and shimmer pass progress (null before it starts). */
data class ExportPose(val lottie: Float, val textMs: Float, val shimmerPass: Float?)

enum class ShareFormat(val label: String) { Jpg("jpg"), Gif("gif"), Mp4("mp4") }

@Composable
private fun ShareButton(
    label: String,
    onClick: () -> Unit,
    enabled: Boolean,
    busy: Boolean,
    ink: Color,
) {
    // Mirrors the web .share-bar button: translucent white pill, 1px ink border, uppercase spaced label.
    val shape = RoundedCornerShape(percent = 50)
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .alpha(if (enabled || busy) 0.55f else 0.3f)
            .clip(shape)
            .background(Color.White.copy(alpha = 0.18f))
            .border(1.dp, ink, shape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 6.dp),
    ) {
        if (busy) {
            CircularProgressIndicator(
                modifier = Modifier.size(14.dp),
                strokeWidth = 2.dp,
                color = ink,
            )
        } else {
            Text(label.uppercase(), color = ink, fontSize = 12.sp, letterSpacing = 1.7.sp)
        }
    }
}
