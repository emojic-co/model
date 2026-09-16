package ing.emojify.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun FeelingBar(feelings: List<String>, active: String?, onPick: (String) -> Unit) {
    if (feelings.isEmpty()) return
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        feelings.forEach { feeling ->
            val isActive = feeling == active
            Text(
                text = feeling,
                modifier = Modifier
                    .clickable { onPick(feeling) }
                    .background(
                        if (isActive) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface,
                        RoundedCornerShape(16.dp),
                    )
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
        }
    }
}
