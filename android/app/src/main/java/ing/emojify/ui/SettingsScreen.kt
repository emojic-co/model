package ing.emojify.ui

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp
import ing.emojify.model.ColorCountPrefs
import ing.emojify.model.ExportSizePrefs
import ing.emojify.model.GifFpsPrefs
import ing.emojify.model.GifSizePrefs
import ing.emojify.model.DebouncePrefs
import ing.emojify.model.EmojiCountPrefs
import ing.emojify.model.LanguagePrefs
import ing.emojify.model.LocalStrings
import ing.emojify.model.ModelUpdatePrefs
import ing.emojify.model.ModelUpdater
import ing.emojify.model.Strings
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private fun statusText(updater: ModelUpdater, strings: Strings): String {
    val installed = updater.installedVersion()
    return when {
        updater.pendingVersion() != null -> strings.t("settings.statusPending")
        installed != null -> strings.t("settings.statusUpdated", mapOf("date" to installed.substringBefore("T")))
        else -> strings.t("settings.statusBundled")
    }
}

private fun openStoreListing(context: Context) {
    val id = context.packageName
    val market = Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=$id"))
    val web = Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=$id"))
    try {
        context.startActivity(market)
    } catch (_: ActivityNotFoundException) {
        try {
            context.startActivity(web)
        } catch (_: ActivityNotFoundException) {
        }
    }
}

@Composable
fun SettingsScreen(onBack: () -> Unit, lang: String, onLangChange: (String) -> Unit) {
    val context = LocalContext.current
    val strings = LocalStrings.current
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
    val colorCountPrefs = remember { context.getSharedPreferences(ColorCountPrefs.FILE, Context.MODE_PRIVATE) }
    var colorCount by remember {
        mutableStateOf(
            colorCountPrefs.getInt(ColorCountPrefs.KEY_COLOR_COUNT, ColorCountPrefs.DEFAULT_COLOR_COUNT)
                .coerceIn(ColorCountPrefs.MIN_COLOR_COUNT, ColorCountPrefs.MAX_COLOR_COUNT),
        )
    }
    val exportSizePrefs = remember { context.getSharedPreferences(ExportSizePrefs.FILE, Context.MODE_PRIVATE) }
    var exportSize by remember {
        mutableStateOf(exportSizePrefs.getInt(ExportSizePrefs.KEY_SIZE_PX, ExportSizePrefs.DEFAULT_SIZE_PX))
    }
    val gifSizePrefs = remember { context.getSharedPreferences(GifSizePrefs.FILE, Context.MODE_PRIVATE) }
    var gifSize by remember {
        mutableStateOf(gifSizePrefs.getInt(GifSizePrefs.KEY_SIZE_PX, GifSizePrefs.DEFAULT_SIZE_PX))
    }
    val gifFpsPrefs = remember { context.getSharedPreferences(GifFpsPrefs.FILE, Context.MODE_PRIVATE) }
    var gifFps by remember {
        mutableStateOf(gifFpsPrefs.getInt(GifFpsPrefs.KEY_FPS, GifFpsPrefs.DEFAULT_FPS))
    }
    val debouncePrefs = remember { context.getSharedPreferences(DebouncePrefs.FILE, Context.MODE_PRIVATE) }
    var debounceMs by remember {
        mutableStateOf(
            debouncePrefs.getInt(DebouncePrefs.KEY_DEBOUNCE_MS, DebouncePrefs.DEFAULT_DEBOUNCE_MS)
                .coerceIn(DebouncePrefs.MIN_DEBOUNCE_MS, DebouncePrefs.MAX_DEBOUNCE_MS),
        )
    }
    var status by remember { mutableStateOf(statusText(updater, strings)) }
    var checking by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    CompositionLocalProvider(
        LocalLayoutDirection provides if (lang == "he") LayoutDirection.Rtl else LayoutDirection.Ltr,
    ) {
    Box(modifier = Modifier.fillMaxSize().safeDrawingPadding().padding(16.dp)) {
        Column {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                IconButton(onClick = onBack) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = strings.t("settings.back"))
                }
                Text(strings.t("settings.title"), style = MaterialTheme.typography.titleLarge)
            }
            Spacer(modifier = Modifier.height(24.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(strings.t("settings.autoUpdate"), modifier = Modifier.weight(1f))
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
                Text(strings.t("settings.wifiOnly"), modifier = Modifier.weight(1f))
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
            Text(strings.t("settings.maxEmojis", mapOf("n" to maxEmojis)))
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
            Text(strings.t("settings.colorCount", mapOf("n" to colorCount)))
            Slider(
                value = colorCount.toFloat(),
                onValueChange = { colorCount = it.roundToInt() },
                onValueChangeFinished = {
                    colorCountPrefs.edit().putInt(ColorCountPrefs.KEY_COLOR_COUNT, colorCount).apply()
                },
                valueRange = ColorCountPrefs.MIN_COLOR_COUNT.toFloat()..ColorCountPrefs.MAX_COLOR_COUNT.toFloat(),
                steps = ColorCountPrefs.MAX_COLOR_COUNT - ColorCountPrefs.MIN_COLOR_COUNT - 1,
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text(strings.t("settings.exportRes"))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (option in ExportSizePrefs.OPTIONS) {
                    val pick = {
                        exportSize = option
                        exportSizePrefs.edit().putInt(ExportSizePrefs.KEY_SIZE_PX, option).apply()
                    }
                    if (option == exportSize) Button(onClick = pick) { Text("${option}x$option") }
                    else OutlinedButton(onClick = pick) { Text("${option}x$option") }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(strings.t("settings.gifRes"))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (option in GifSizePrefs.OPTIONS) {
                    val pick = {
                        gifSize = option
                        gifSizePrefs.edit().putInt(GifSizePrefs.KEY_SIZE_PX, option).apply()
                    }
                    if (option == gifSize) Button(onClick = pick) { Text("${option}x$option") }
                    else OutlinedButton(onClick = pick) { Text("${option}x$option") }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(strings.t("settings.gifFps"))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (option in GifFpsPrefs.OPTIONS) {
                    val pick = {
                        gifFps = option
                        gifFpsPrefs.edit().putInt(GifFpsPrefs.KEY_FPS, option).apply()
                    }
                    if (option == gifFps) Button(onClick = pick) { Text("$option") }
                    else OutlinedButton(onClick = pick) { Text("$option") }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))
            Text(strings.t("settings.debounce", mapOf("ms" to debounceMs)))
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
            Spacer(modifier = Modifier.height(24.dp))
            Text(strings.t("settings.language"))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (code in LanguagePrefs.SUPPORTED) {
                    val name = LanguagePrefs.NAMES.getValue(code)
                    if (code == lang) Button(onClick = { onLangChange(code) }) { Text(name) }
                    else OutlinedButton(onClick = { onLangChange(code) }) { Text(name) }
                }
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
                        status = statusText(updater, strings)
                        checking = false
                    }
                },
            ) {
                Text(strings.t(if (checking) "settings.checking" else "settings.checkNow"))
            }
            Spacer(modifier = Modifier.height(24.dp))
            OutlinedButton(onClick = { openStoreListing(context) }) {
                Text(strings.t("settings.rate"))
            }
        }
    }
    }
}
