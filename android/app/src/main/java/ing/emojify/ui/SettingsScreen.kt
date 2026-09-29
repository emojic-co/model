package ing.emojify.ui

import android.content.Context
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import ing.emojify.model.DebouncePrefs
import ing.emojify.model.EmojiCountPrefs
import ing.emojify.model.ModelUpdatePrefs
import ing.emojify.model.ModelUpdater
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private fun statusText(updater: ModelUpdater): String {
    val installed = updater.installedVersion()
    return when {
        updater.pendingVersion() != null -> "Update available, waiting for Wi-Fi"
        installed != null -> "Model updated ${installed.substringBefore("T")}"
        else -> "Using bundled model"
    }
}

@Composable
fun SettingsScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences(ModelUpdatePrefs.FILE, Context.MODE_PRIVATE) }
    val emojiCountPrefs = remember { context.getSharedPreferences(EmojiCountPrefs.FILE, Context.MODE_PRIVATE) }
    val updater = remember { ModelUpdater(context) }
    var autoUpdate by remember {
        mutableStateOf(prefs.getBoolean(ModelUpdatePrefs.KEY_AUTO_UPDATE, ModelUpdatePrefs.DEFAULT_AUTO_UPDATE))
    }
    var wifiOnly by remember {
        mutableStateOf(prefs.getBoolean(ModelUpdatePrefs.KEY_WIFI_ONLY, ModelUpdatePrefs.DEFAULT_WIFI_ONLY))
    }
    var maxEmojis by remember {
        mutableStateOf(
            emojiCountPrefs.getInt(EmojiCountPrefs.KEY_MAX_EMOJIS, EmojiCountPrefs.DEFAULT_MAX_EMOJIS)
                .coerceIn(EmojiCountPrefs.MIN_MAX_EMOJIS, EmojiCountPrefs.MAX_MAX_EMOJIS),
        )
    }
    val debouncePrefs = remember { context.getSharedPreferences(DebouncePrefs.FILE, Context.MODE_PRIVATE) }
    var debounceMs by remember {
        mutableStateOf(
            debouncePrefs.getInt(DebouncePrefs.KEY_DEBOUNCE_MS, DebouncePrefs.DEFAULT_DEBOUNCE_MS)
                .coerceIn(DebouncePrefs.MIN_DEBOUNCE_MS, DebouncePrefs.MAX_DEBOUNCE_MS),
        )
    }
    var status by remember { mutableStateOf(statusText(updater)) }
    var checking by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Box(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
                Text("Settings", style = MaterialTheme.typography.titleLarge)
            }
            Spacer(modifier = Modifier.height(24.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Auto-update model", modifier = Modifier.weight(1f))
                Switch(
                    checked = autoUpdate,
                    onCheckedChange = {
                        autoUpdate = it
                        prefs.edit().putBoolean(ModelUpdatePrefs.KEY_AUTO_UPDATE, it).apply()
                    },
                )
            }
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Wi-Fi only", modifier = Modifier.weight(1f))
                Switch(
                    checked = wifiOnly,
                    enabled = autoUpdate,
                    onCheckedChange = {
                        wifiOnly = it
                        prefs.edit().putBoolean(ModelUpdatePrefs.KEY_WIFI_ONLY, it).apply()
                    },
                )
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text("Max emojis shown: $maxEmojis")
            Slider(
                value = maxEmojis.toFloat(),
                onValueChange = { maxEmojis = it.roundToInt() },
                onValueChangeFinished = {
                    emojiCountPrefs.edit().putInt(EmojiCountPrefs.KEY_MAX_EMOJIS, maxEmojis).apply()
                },
                valueRange = EmojiCountPrefs.MIN_MAX_EMOJIS.toFloat()..EmojiCountPrefs.MAX_MAX_EMOJIS.toFloat(),
                steps = EmojiCountPrefs.MAX_MAX_EMOJIS - EmojiCountPrefs.MIN_MAX_EMOJIS - 1,
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text("Typing delay before card updates: $debounceMs ms")
            Slider(
                value = debounceMs.toFloat(),
                onValueChange = {
                    debounceMs = (it / DebouncePrefs.STEP_MS).roundToInt() * DebouncePrefs.STEP_MS
                },
                onValueChangeFinished = {
                    debouncePrefs.edit().putInt(DebouncePrefs.KEY_DEBOUNCE_MS, debounceMs).apply()
                },
                valueRange = DebouncePrefs.MIN_DEBOUNCE_MS.toFloat()..DebouncePrefs.MAX_DEBOUNCE_MS.toFloat(),
                steps = (DebouncePrefs.MAX_DEBOUNCE_MS - DebouncePrefs.MIN_DEBOUNCE_MS) / DebouncePrefs.STEP_MS - 1,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(status)
            Spacer(modifier = Modifier.height(12.dp))
            Button(
                enabled = !checking,
                onClick = {
                    checking = true
                    scope.launch {
                        withContext(Dispatchers.IO) { updater.run(wifiOnly = wifiOnly, force = true) }
                        status = statusText(updater)
                        checking = false
                    }
                },
            ) {
                Text(if (checking) "Checking…" else "Check now")
            }
        }
    }
}
