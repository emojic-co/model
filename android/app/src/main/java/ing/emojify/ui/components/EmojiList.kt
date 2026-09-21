package ing.emojify.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ing.emojify.model.EmojiScore

@Composable
fun EmojiList(items: List<EmojiScore>?, active: String?, state: LazyListState = rememberLazyListState(), onPick: (String) -> Unit) {
    if (items.isNullOrEmpty()) return
    LazyRow(state = state, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(items, key = { it.emoji }) { item ->
            val isActive = item.emoji == active
            Text(
                text = item.emoji,
                style = MaterialTheme.typography.headlineSmall,
                modifier = Modifier
                    .clickable { onPick(item.emoji) }
                    .background(
                        if (isActive) MaterialTheme.colorScheme.secondaryContainer else MaterialTheme.colorScheme.surface,
                        CircleShape,
                    )
                    .padding(8.dp),
            )
        }
    }
}
