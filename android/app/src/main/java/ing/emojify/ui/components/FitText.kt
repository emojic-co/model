package ing.emojify.ui.components

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextDirection
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.sp

@Composable
fun rememberFitFontSizeSp(
    text: String,
    fontFamily: FontFamily,
    fontWeight: FontWeight = FontWeight.Normal,
    fontStyle: FontStyle = FontStyle.Normal,
    letterSpacing: TextUnit = TextUnit.Unspecified,
    maxWidthPx: Int,
    maxHeightPx: Int,
    minSp: Float,
    maxSp: Float,
    lineHeightMultiplier: Float = 1.2f,
): Float {
    val textMeasurer = rememberTextMeasurer()
    return remember(text, fontFamily, fontWeight, fontStyle, letterSpacing, maxWidthPx, maxHeightPx, minSp, maxSp, lineHeightMultiplier) {
        if (maxWidthPx <= 0 || maxHeightPx <= 0 || text.isEmpty() || minSp >= maxSp) return@remember minSp.coerceAtLeast(1f)

        fun fits(sizeSp: Float): Boolean {
            val result = textMeasurer.measure(
                text = text,
                style = TextStyle(
                    fontSize = sizeSp.sp,
                    lineHeight = (sizeSp * lineHeightMultiplier).sp,
                    fontFamily = fontFamily,
                    fontWeight = fontWeight,
                    fontStyle = fontStyle,
                    letterSpacing = letterSpacing,
                    textDirection = TextDirection.Content,
                ),
                constraints = Constraints(maxWidth = maxWidthPx, maxHeight = maxHeightPx),
                softWrap = true,
            )
            return !result.hasVisualOverflow
        }

        var lo = minSp
        var hi = maxSp
        if (!fits(lo)) return@remember lo
        repeat(12) {
            val mid = (lo + hi) / 2f
            if (fits(mid)) lo = mid else hi = mid
        }
        lo
    }
}
