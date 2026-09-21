package ing.emojify

import android.content.Context
import android.content.res.AssetManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import ing.emojify.model.Meta
import ing.emojify.model.ModelUpdatePrefs
import ing.emojify.model.ModelUpdater
import ing.emojify.model.OnnxPredictor
import ing.emojify.ui.MainScreen
import ing.emojify.ui.SettingsScreen
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json

private fun loadBundledModel(assets: AssetManager): Pair<String, ByteArray> {
    val metaJson = assets.open("meta.json").bufferedReader().use { it.readText() }
    val modelBytes = assets.open("model.onnx").use { it.readBytes() }
    return metaJson to modelBytes
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val updater = ModelUpdater(applicationContext)
        val (metaJson, modelBytes) = if (updater.hasCachedModel()) {
            try {
                updater.cachedMetaFile.readText() to updater.cachedModelFile.readBytes()
            } catch (_: Exception) {
                loadBundledModel(assets)
            }
        } else {
            loadBundledModel(assets)
        }
        val meta = Json.decodeFromString(Meta.serializer(), metaJson)
        val predictor = OnnxPredictor(modelBytes, meta)

        val prefs = getSharedPreferences(ModelUpdatePrefs.FILE, Context.MODE_PRIVATE)
        val wifiOnly = prefs.getBoolean(ModelUpdatePrefs.KEY_WIFI_ONLY, ModelUpdatePrefs.DEFAULT_WIFI_ONLY)
        CoroutineScope(Dispatchers.IO).launch { updater.run(wifiOnly = wifiOnly) }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    var showSettings by remember { mutableStateOf(false) }
                    if (showSettings) {
                        SettingsScreen(onBack = { showSettings = false })
                    } else {
                        MainScreen(meta = meta, predictor = predictor, onSettingsClick = { showSettings = true })
                    }
                }
            }
        }
    }
}
