package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.compose.ui.graphics.layer.drawLayer
import androidx.compose.ui.graphics.rememberGraphicsLayer
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import ing.emojify.app.R
import ing.emojify.app.model.Palette
import ing.emojify.app.model.patternTint
import ing.emojify.app.model.resolveFeeling

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
    colors: Palette,
    onCopy: () -> Unit,
    onShare: () -> Unit,
    onCaptureReady: ((suspend () -> android.graphics.Bitmap) -> Unit)? = null,
) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    val style = resolveFeeling(feeling)
    val tint = Color(android.graphics.Color.parseColor(patternTint(colors.bg1, colors.bg2)))
    val fontFamily = FontFamily(Font(googleFont = GoogleFont(style.fontName), fontProvider = fontProvider))
    val displayText = if (style.uppercase) text.uppercase() else text
    val graphicsLayer = rememberGraphicsLayer()
    LaunchedEffect(onCaptureReady) {
        onCaptureReady?.invoke { graphicsLayer.toImageBitmap().asAndroidBitmap() }
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .drawWithContent {
                graphicsLayer.record { this@drawWithContent.drawContent() }
                drawLayer(graphicsLayer)
            },
    ) {
        PatternBackground(cluster = style.cluster, tint = tint, modifier = Modifier.matchParentSize())
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(Brush.linearGradient(listOf(bg1.copy(alpha = 0.85f), bg2.copy(alpha = 0.85f))))
                .padding(32.dp),
        ) {
            val emojiBounce = rememberEmojiBounce(style.emojiMotif, style.emojiMs)
            val entranceScale = rememberEntranceScale(style.entranceMotif, style.entranceMs, key = text)
            Column {
                Text(
                    text = emoji,
                    style = MaterialTheme.typography.displayLarge,
                    color = textColor,
                    modifier = Modifier.offset(y = emojiBounce.value.dp),
                )
                Text(
                    text = displayText.ifBlank { "What's on your mind?" },
                    color = textColor,
                    fontFamily = fontFamily,
                    fontWeight = if (style.bold) FontWeight.Bold else FontWeight.Normal,
                    fontStyle = if (style.italic) FontStyle.Italic else FontStyle.Normal,
                    letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
                    modifier = Modifier.scale(entranceScale.value),
                )
                Row {
                    TextButton(onClick = onShare) { Text("share", color = textColor) }
                    TextButton(onClick = onCopy) { Text("copy", color = textColor) }
                }
            }
        }
    }
}
