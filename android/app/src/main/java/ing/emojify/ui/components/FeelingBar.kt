package ing.emojify.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.TextUnitType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ing.emojify.R
import ing.emojify.model.resolveFeeling

private val SWATCH_BG = Color(0xFFE7E4DF)
private val SWATCH_INK = Color(0xFF33312E)
val SWATCH_MIN_SIZE = 84.dp
val SWATCH_MAX_SIZE = 160.dp
private val SWATCH_PADDING = 10.dp

private val feelingFontProvider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

@Composable
private fun FeelingSwatch(feeling: String, isActive: Boolean, swatchSize: androidx.compose.ui.unit.Dp, onPick: (String) -> Unit) {
    val style = resolveFeeling(feeling)
    val fontFamily = FontFamily(Font(googleFont = GoogleFont(style.fontName), fontProvider = feelingFontProvider))
    val displayText = if (style.uppercase) feeling.uppercase() else feeling
    val density = LocalDensity.current
    val innerPx = with(density) { (swatchSize - SWATCH_PADDING * 2).toPx() }.toInt()
    val fitSp = rememberFitFontSizeSp(
        text = displayText,
        fontFamily = fontFamily,
        fontWeight = if (style.bold) androidx.compose.ui.text.font.FontWeight.Bold else androidx.compose.ui.text.font.FontWeight.Normal,
        fontStyle = if (style.italic) androidx.compose.ui.text.font.FontStyle.Italic else androidx.compose.ui.text.font.FontStyle.Normal,
        letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
        maxWidthPx = innerPx,
        maxHeightPx = innerPx,
        minSp = 9f,
        maxSp = 18f,
    )
    Box(
        contentAlignment = Alignment.Center,
        modifier = Modifier
            .size(swatchSize)
            .aspectRatio(1f)
            .clip(RoundedCornerShape(14.dp))
            .background(SWATCH_BG)
            .border(
                width = if (isActive) 2.dp else 0.dp,
                color = if (isActive) MaterialTheme.colorScheme.onSurface else Color.Transparent,
                shape = RoundedCornerShape(14.dp),
            )
            .clickable { onPick(feeling) },
    ) {
        PatternBackground(cluster = style.cluster, tint = SWATCH_INK, opacity = 0.14f, modifier = Modifier.matchParentSize())
        Text(
            text = displayText,
            color = SWATCH_INK,
            fontFamily = fontFamily,
            fontSize = fitSp.sp,
            lineHeight = (fitSp * 1.2f).sp,
            textAlign = TextAlign.Center,
            fontWeight = if (style.bold) androidx.compose.ui.text.font.FontWeight.Bold else androidx.compose.ui.text.font.FontWeight.Normal,
            fontStyle = if (style.italic) androidx.compose.ui.text.font.FontStyle.Italic else androidx.compose.ui.text.font.FontStyle.Normal,
            letterSpacing = style.letterSpacingEm?.let { TextUnit(it, TextUnitType.Em) } ?: TextUnit.Unspecified,
            modifier = Modifier.padding(SWATCH_PADDING),
        )
    }
}

@Composable
fun FeelingBar(
    feelings: List<String>,
    active: String?,
    swatchSize: androidx.compose.ui.unit.Dp = SWATCH_MIN_SIZE,
    onPick: (String) -> Unit,
) {
    if (feelings.isEmpty()) return
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(feelings, key = { it }) { feeling ->
            FeelingSwatch(feeling = feeling, isActive = feeling == active, swatchSize = swatchSize, onPick = onPick)
        }
    }
}
