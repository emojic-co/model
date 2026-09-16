package ing.emojify.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import ing.emojify.model.Palette

@Composable
fun ColorBar(palettes: List<Palette>, active: Int, onPick: (Int) -> Unit) {
    if (palettes.size <= 1) return
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        palettes.forEachIndexed { index, palette ->
            val isActive = index == active
            val bg1 = Color(android.graphics.Color.parseColor(palette.bg1))
            val bg2 = Color(android.graphics.Color.parseColor(palette.bg2))
            val textColor = Color(android.graphics.Color.parseColor(palette.textColor))
            Box(
                contentAlignment = Alignment.Center,
                modifier = Modifier
                    .weight(1f)
                    .aspectRatio(1f)
                    .clickable { onPick(index) }
                    .background(Brush.linearGradient(listOf(bg1, bg2)), RoundedCornerShape(12.dp))
                    .border(
                        width = if (isActive) 2.dp else 0.dp,
                        color = if (isActive) MaterialTheme.colorScheme.onSurface else Color.Transparent,
                        shape = RoundedCornerShape(12.dp),
                    ),
            ) {
                Text("Aa", color = textColor, fontWeight = FontWeight.SemiBold)
            }
        }
    }
}
