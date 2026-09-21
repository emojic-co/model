package ing.emojify.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.scale
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
import ing.emojify.model.patternTint
import ing.emojify.model.resolveFeeling

private val fontProvider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

// Proportions mirror web/src/styles.css's `.card` cqw-based sizing (card-emoji: 32cqw,
// .card { padding: 7% }, useFitText's { min: 5, max: 13 } cqw range for the card text).
private const val EMOJI_SIZE_RATIO = 0.32f
private const val CARD_PADDING_RATIO = 0.07f
private const val TEXT_MIN_SIZE_RATIO = 0.05f
private const val TEXT_MAX_SIZE_RATIO = 0.13f
private const val PATTERN_TILE_RATIO = 0.28f

@Composable
fun Card(
    text: String,
    emoji: String,
    feeling: String?,
    lang: String?,
    colors: Palette,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onEmojiCycle: (Int) -> Unit,
    onFeelingCycle: (Int) -> Unit,
    onCaptureReady: ((suspend () -> android.graphics.Bitmap) -> Unit)? = null,
) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    val style = resolveFeeling(feeling, lang)
    val tint = Color(android.graphics.Color.parseColor(patternTint(colors.bg1, colors.bg2)))
    val fontFamily = FontFamily(Font(googleFont = GoogleFont(style.fontName), fontProvider = fontProvider))
    val displayText = if (style.uppercase) text.uppercase() else text
    val graphicsLayer = rememberGraphicsLayer()
    LaunchedEffect(onCaptureReady) {
        onCaptureReady?.invoke { graphicsLayer.toImageBitmap().asAndroidBitmap() }
    }

    BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
        val cardWidthDp = maxWidth
        val emojiSizeSp = cardWidthDp.value * EMOJI_SIZE_RATIO
        val cardPadding = cardWidthDp * CARD_PADDING_RATIO
        val textMinSp = cardWidthDp.value * TEXT_MIN_SIZE_RATIO
        val textMaxSp = cardWidthDp.value * TEXT_MAX_SIZE_RATIO
        val density = LocalDensity.current
        val patternTilePx = with(density) { (cardWidthDp * PATTERN_TILE_RATIO).toPx() }.toInt().coerceAtLeast(1)

        Column {
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
                PatternBackground(
                    cluster = style.cluster,
                    tint = tint,
                    modifier = Modifier.matchParentSize(),
                    tilePx = patternTilePx,
                )
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(cardPadding),
                ) {
                    val emojiBounce = rememberEmojiBounce(style.emojiMotif, style.emojiMs)
                    val entranceScale = rememberEntranceScale(style.entranceMotif, style.entranceMs, key = text)
                    Column(modifier = Modifier.fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = emoji,
                            fontSize = emojiSizeSp.sp,
                            color = textColor,
                            modifier = Modifier.offset(y = emojiBounce.value.dp),
                        )
                        Box(
                            modifier = Modifier.weight(1f).fillMaxWidth(),
                            contentAlignment = Alignment.Center,
                        ) {
                            BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                                val density = LocalDensity.current
                                val maxWidthPx = with(density) { maxWidth.toPx() }.toInt()
                                val maxHeightPx = with(density) { maxHeight.toPx() }.toInt()
                                val fitSp = rememberFitFontSizeSp(
                                    text = displayText.ifBlank { "What's on your mind?" },
                                    fontFamily = fontFamily,
                                    fontWeight = if (style.bold) FontWeight.Bold else FontWeight.Normal,
                                    fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                    letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    maxWidthPx = maxWidthPx,
                                    maxHeightPx = maxHeightPx,
                                    minSp = textMinSp,
                                    maxSp = textMaxSp,
                                )
                                Text(
                                    text = displayText.ifBlank { "What's on your mind?" },
                                    style = TextStyle(textDirection = TextDirection.Content),
                                    color = textColor,
                                    textAlign = TextAlign.Center,
                                    fontFamily = fontFamily,
                                    fontSize = fitSp.sp,
                                    lineHeight = (fitSp * 1.2f).sp,
                                    fontWeight = if (style.bold) FontWeight.Bold else FontWeight.Normal,
                                    fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                    letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    modifier = Modifier.fillMaxWidth().scale(entranceScale.value),
                                )
                            }
                        }
                    }
                }
            }
            Row {
                TextButton(onClick = onShare) { Text("share") }
                TextButton(onClick = onCopy) { Text("copy") }
            }
        }
    }
}
