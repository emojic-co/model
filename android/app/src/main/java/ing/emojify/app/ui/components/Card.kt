package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ing.emojify.app.model.Palette

@Composable
fun Card(text: String, emoji: String, feeling: String?, colors: Palette, onCopy: () -> Unit, onShare: () -> Unit) {
    val bg1 = Color(android.graphics.Color.parseColor(colors.bg1))
    val bg2 = Color(android.graphics.Color.parseColor(colors.bg2))
    val textColor = Color(android.graphics.Color.parseColor(colors.textColor))
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Brush.linearGradient(listOf(bg1, bg2)), RoundedCornerShape(16.dp))
            .padding(32.dp),
    ) {
        Column {
            Text(text = emoji, style = MaterialTheme.typography.displayLarge, color = textColor)
            Text(text = text.ifBlank { "What's on your mind?" }, color = textColor)
        }
    }
}
