package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.Palette

@Composable
fun ColorBar(palettes: List<Palette>, active: Int, onPick: (Int) -> Unit) {
    if (palettes.size <= 1) return
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        palettes.forEachIndexed { index, palette ->
            val isActive = index == active
            androidx.compose.foundation.layout.Box(
                modifier = Modifier
                    .size(28.dp)
                    .clickable { onPick(index) }
                    .background(Color(android.graphics.Color.parseColor(palette.bg1)), CircleShape)
                    .border(
                        width = if (isActive) 2.dp else 0.dp,
                        color = MaterialTheme.colorScheme.primary,
                        shape = CircleShape,
                    ),
            )
        }
    }
}
