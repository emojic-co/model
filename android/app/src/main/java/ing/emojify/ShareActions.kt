package ing.emojify

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.core.content.FileProvider
import java.io.File
import ing.emojify.model.ShareLinkPrefs
import java.io.FileOutputStream
import kotlin.random.Random

private fun cardsDir(context: Context) = File(context.cacheDir, "cards").apply { mkdirs() }

private fun shouldAddLink(context: Context): Boolean {
    val prefs = context.getSharedPreferences(ShareLinkPrefs.FILE, Context.MODE_PRIVATE)
    return prefs.getBoolean(ShareLinkPrefs.KEY_ENABLED, ShareLinkPrefs.DEFAULT_ENABLED) &&
        Random.nextInt(ShareLinkPrefs.ONE_IN) == 0
}

private fun send(context: Context, file: File, mime: String) {
    val uri = FileProvider.getUriForFile(context, "ing.emojify.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = mime
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        if (shouldAddLink(context)) putExtra(Intent.EXTRA_TEXT, ShareLinkPrefs.URL)
    }
    context.startActivity(Intent.createChooser(intent, null))
}

fun shareCardJpg(context: Context, bitmap: Bitmap, sizePx: Int) {
    val readable = if (bitmap.config == Bitmap.Config.HARDWARE) bitmap.copy(Bitmap.Config.ARGB_8888, false) else bitmap
    val scaled = Bitmap.createScaledBitmap(readable, sizePx, sizePx, true)
    val file = File(cardsDir(context), "card.jpg")
    FileOutputStream(file).use { scaled.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    send(context, file, "image/jpeg")
}

fun cardGifFile(context: Context): File = File(cardsDir(context), "card.gif")

fun shareCardGif(context: Context, file: File) = send(context, file, "image/gif")

fun cardMp4File(context: Context): File = File(cardsDir(context), "card.mp4")

fun shareCardMp4(context: Context, file: File) = send(context, file, "video/mp4")
