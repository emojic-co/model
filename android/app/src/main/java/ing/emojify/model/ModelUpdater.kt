package ing.emojify.model

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.serialization.json.Json

object ModelUpdatePrefs {
    const val FILE = "model_updater"
    const val KEY_AUTO_UPDATE = "auto_update_enabled"
    const val KEY_WIFI_ONLY = "wifi_only"
    const val DEFAULT_AUTO_UPDATE = true
    const val DEFAULT_WIFI_ONLY = true
}

private const val BASE_URL = "https://emojify.ing/"
private const val KEY_LAST_CHECKED_AT = "last_checked_at"
private const val KEY_INSTALLED_VERSION = "installed_version"
private const val KEY_PENDING_VERSION = "pending_version"
private const val CHECK_INTERVAL_MS = 7L * 24 * 60 * 60 * 1000

class ModelUpdater(private val context: Context) {
    private val prefs = context.getSharedPreferences(ModelUpdatePrefs.FILE, Context.MODE_PRIVATE)
    private val modelDir = File(context.filesDir, "model")

    val cachedModelFile = File(modelDir, "model.onnx")
    val cachedMetaFile = File(modelDir, "meta.json")

    fun hasCachedModel(): Boolean = cachedModelFile.exists() && cachedMetaFile.exists()

    fun installedVersion(): String? = prefs.getString(KEY_INSTALLED_VERSION, null)

    fun pendingVersion(): String? = prefs.getString(KEY_PENDING_VERSION, null)

    fun run(wifiOnly: Boolean, force: Boolean = false) {
        if (!force && !prefs.getBoolean(ModelUpdatePrefs.KEY_AUTO_UPDATE, ModelUpdatePrefs.DEFAULT_AUTO_UPDATE)) return
        checkForUpdate(force)
        downloadIfPending(wifiOnly)
    }

    private fun checkForUpdate(force: Boolean) {
        val lastChecked = prefs.getLong(KEY_LAST_CHECKED_AT, 0)
        val now = System.currentTimeMillis()
        if (!force && now - lastChecked < CHECK_INTERVAL_MS) return
        try {
            val metaText = URL(BASE_URL + "meta.json").readText()
            val remoteVersion = Json.decodeFromString(Meta.serializer(), metaText).exported_at
            prefs.edit().putLong(KEY_LAST_CHECKED_AT, now).apply()
            if (remoteVersion != null && remoteVersion != installedVersion()) {
                prefs.edit().putString(KEY_PENDING_VERSION, remoteVersion).apply()
            }
        } catch (_: Exception) {
        }
    }

    private fun downloadIfPending(wifiOnly: Boolean) {
        val pending = pendingVersion() ?: return
        if (wifiOnly && !isOnWifi()) return
        val modelTmp = File(modelDir, "model.onnx.tmp")
        val metaTmp = File(modelDir, "meta.json.tmp")
        try {
            modelDir.mkdirs()
            downloadTo(BASE_URL + "model.onnx", modelTmp)
            downloadTo(BASE_URL + "meta.json", metaTmp)
            if (modelTmp.length() == 0L) throw java.io.IOException("empty model download")
            Json.decodeFromString(Meta.serializer(), metaTmp.readText())
            modelTmp.copyTo(cachedModelFile, overwrite = true)
            metaTmp.copyTo(cachedMetaFile, overwrite = true)
            prefs.edit()
                .putString(KEY_INSTALLED_VERSION, pending)
                .remove(KEY_PENDING_VERSION)
                .apply()
        } catch (_: Exception) {
        } finally {
            modelTmp.delete()
            metaTmp.delete()
        }
    }

    private fun downloadTo(urlStr: String, dst: File) {
        val conn = URL(urlStr).openConnection() as HttpURLConnection
        conn.connectTimeout = 15_000
        conn.readTimeout = 30_000
        conn.inputStream.use { input -> dst.outputStream().use { output -> input.copyTo(output) } }
        conn.disconnect()
    }

    private fun isOnWifi(): Boolean {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(network) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
    }
}
