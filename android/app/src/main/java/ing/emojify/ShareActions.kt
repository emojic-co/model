package ing.emojify

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

private fun cardsDir(context: Context) = File(context.cacheDir, "cards").apply { mkdirs() }

private fun send(context: Context, file: File, mime: String) {
    val uri = FileProvider.getUriForFile(context, "ing.emojify.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = mime
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
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
