package ing.emojify.app.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun MainScreen() {
    var text by remember { mutableStateOf("") }
    Column(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        TextField(
            value = text,
            onValueChange = { text = it },
            placeholder = { Text("type at least 3 characters…") },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "🙂", style = MaterialTheme.typography.displayLarge)
        Text(text = "What's on your mind?")
    }
}
