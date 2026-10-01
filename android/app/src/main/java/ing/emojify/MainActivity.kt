package ing.emojify

import android.content.Context
import android.content.res.AssetManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import ing.emojify.model.LanguagePrefs
import ing.emojify.model.LocalStrings
import ing.emojify.model.Meta
import ing.emojify.model.ModelUpdatePrefs
import ing.emojify.model.ModelUpdater
import ing.emojify.model.OnnxPredictor
import ing.emojify.model.Strings
import ing.emojify.model.Styles
import ing.emojify.model.StyleUpdater
import ing.emojify.model.parseStyleFile
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

private fun loadBundledStyle(assets: AssetManager): String =
    assets.open("style.yml").bufferedReader().use { it.readText() }

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

        val styleUpdater = StyleUpdater(applicationContext)
        // A downloaded style.yml only wins if it is newer than the one bundled in this build,
        // so a stale cache can never hide features the installed app ships with.
        val bundledStyle = parseStyleFile(loadBundledStyle(assets))
        val cachedStyle = if (styleUpdater.hasCachedStyle()) {
            try {
                parseStyleFile(styleUpdater.cachedStyleFile.readText())
            } catch (_: Exception) {
                null
            }
        } else {
            null
        }
        val chosenStyle = if (cachedStyle != null && cachedStyle.exportedAt > bundledStyle.exportedAt) cachedStyle else bundledStyle
        Styles.init(chosenStyle)
        // A cached style.yml from before the i18n section existed falls back to the bundled strings.
        val i18n = (chosenStyle.i18n ?: bundledStyle.i18n)!!
        val langPrefs = getSharedPreferences(LanguagePrefs.FILE, Context.MODE_PRIVATE)

        val prefs = getSharedPreferences(ModelUpdatePrefs.FILE, Context.MODE_PRIVATE)
        val wifiOnly = prefs.getBoolean(ModelUpdatePrefs.KEY_WIFI_ONLY, ModelUpdatePrefs.DEFAULT_WIFI_ONLY)
        CoroutineScope(Dispatchers.IO).launch { updater.run(wifiOnly = wifiOnly) }
        CoroutineScope(Dispatchers.IO).launch { styleUpdater.run(wifiOnly = wifiOnly) }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    var lang by remember {
                        mutableStateOf(
                            LanguagePrefs.resolve(
                                langPrefs.getString(LanguagePrefs.KEY_LANG, null),
                                java.util.Locale.getDefault().language,
                            ),
                        )
                    }
                    val strings = remember(lang) { Strings(lang, i18n) }
                    CompositionLocalProvider(LocalStrings provides strings) {
                        var showSettings by remember { mutableStateOf(false) }
                        BackHandler(enabled = showSettings) { showSettings = false }
                        if (showSettings) {
                            SettingsScreen(
                                onBack = { showSettings = false },
                                lang = lang,
                                onLangChange = {
                                    lang = it
                                    langPrefs.edit().putString(LanguagePrefs.KEY_LANG, it).apply()
                                },
                            )
                        } else {
                            MainScreen(meta = meta, predictor = predictor, onSettingsClick = { showSettings = true })
                        }
                    }
                }
            }
        }
    }
}
