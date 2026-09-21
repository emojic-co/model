package ing.emojify.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FloatingActionButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
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
import ing.emojify.model.Styles
import ing.emojify.model.patternTint
import ing.emojify.model.resolveFeeling

private val fontProvider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

@Composable
fun Card(
    text: String,
    emoji: String,
    feeling: String?,
    lang: String?,
    colors: Palette,
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
                                    fontWeight = FontWeight(style.fontWeight),
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
                                    fontWeight = FontWeight(style.fontWeight),
                                    fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                                    letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                                    modifier = Modifier.fillMaxWidth().scale(entranceScale.value),
                                )
                            }
                        }
                    }
                }
            }
            ExtendedFloatingActionButton(
                onClick = onShare,
                icon = { Icon(Icons.Filled.Share, contentDescription = null) },
                text = { Text("Share") },
                containerColor = Color.White.copy(alpha = 0.75f),
                contentColor = Color.Black.copy(alpha = 0.8f),
                elevation = FloatingActionButtonDefaults.elevation(defaultElevation = 1.dp, pressedElevation = 1.dp),
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .offset(y = (-16).dp),
            )
        }
    }
}
