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
import ing.emojify.model.ModelUpdatePrefs
import ing.emojify.model.ModelUpdater
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
    val updater = remember { ModelUpdater(context) }
    var autoUpdate by remember {
        mutableStateOf(prefs.getBoolean(ModelUpdatePrefs.KEY_AUTO_UPDATE, ModelUpdatePrefs.DEFAULT_AUTO_UPDATE))
    }
    var wifiOnly by remember {
        mutableStateOf(prefs.getBoolean(ModelUpdatePrefs.KEY_WIFI_ONLY, ModelUpdatePrefs.DEFAULT_WIFI_ONLY))
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
