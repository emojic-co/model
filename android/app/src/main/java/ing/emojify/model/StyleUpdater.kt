package ing.emojify.model

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

private const val BASE_URL = "https://emojify.ing/"
private const val KEY_LAST_CHECKED_AT = "style_last_checked_at"
private const val KEY_INSTALLED_VERSION = "style_installed_version"
private const val KEY_PENDING_VERSION = "style_pending_version"
private const val CHECK_INTERVAL_MS = 7L * 24 * 60 * 60 * 1000

class StyleUpdater(private val context: Context) {
    private val prefs = context.getSharedPreferences(ModelUpdatePrefs.FILE, Context.MODE_PRIVATE)
    private val styleDir = File(context.filesDir, "style")

    val cachedStyleFile = File(styleDir, "style.yml")

    fun hasCachedStyle(): Boolean = cachedStyleFile.exists()

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
            val yaml = URL(BASE_URL + "style.yml").readText()
            val remoteVersion = parseStyleFile(yaml).exportedAt
            prefs.edit().putLong(KEY_LAST_CHECKED_AT, now).apply()
            if (remoteVersion != installedVersion()) {
                prefs.edit().putString(KEY_PENDING_VERSION, remoteVersion).apply()
            }
        } catch (_: Exception) {
        }
    }

    private fun downloadIfPending(wifiOnly: Boolean) {
        val pending = pendingVersion() ?: return
        if (wifiOnly && !isOnWifi()) return
        val tmp = File(styleDir, "style.yml.tmp")
        try {
            styleDir.mkdirs()
            downloadTo(BASE_URL + "style.yml", tmp)
            if (tmp.length() == 0L) throw IOException("empty style download")
            parseStyleFile(tmp.readText())
            tmp.copyTo(cachedStyleFile, overwrite = true)
            prefs.edit()
                .putString(KEY_INSTALLED_VERSION, pending)
                .remove(KEY_PENDING_VERSION)
                .apply()
        } catch (_: Exception) {
        } finally {
            tmp.delete()
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
