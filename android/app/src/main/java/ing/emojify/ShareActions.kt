package ing.emojify

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream

private fun writeCardPng(context: Context, bitmap: Bitmap): android.net.Uri {
    val dir = File(context.cacheDir, "cards").apply { mkdirs() }
    val file = File(dir, "card.png")
    FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }
    return FileProvider.getUriForFile(context, "ing.emojify.fileprovider", file)
}

fun shareCard(context: Context, bitmap: Bitmap) {
    val uri = writeCardPng(context, bitmap)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/png"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, null))
}

fun copyCardToClipboard(context: Context, bitmap: Bitmap) {
    val uri = writeCardPng(context, bitmap)
    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
    clipboard.setPrimaryClip(ClipData.newUri(context.contentResolver, "emojify card", uri))
}
