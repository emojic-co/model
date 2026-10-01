package ing.emojify.model

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

// Animated Noto emoji (Lottie, CC BY 4.0) bundled under assets/noto/, synced from web/public/noto/
// by `bun run fetch-noto-lottie`. Emojis missing from index.json have no animated clone.
object NotoLottie {
    private const val DIR = "noto"

    @Volatile private var index: Map<String, String>? = null

    fun assetPath(context: Context, emoji: String): String? {
        val idx = index ?: synchronized(this) {
            index ?: load(context).also { index = it }
        }
        return idx[emoji]?.let { "$DIR/$it.json" }
    }

    /** Lottie clip length in ms, or 0.0 when the emoji has no clone (it plays the fallback spring instead). */
    suspend fun loadEmojiLoopMsOf(context: Context, emoji: String): Double {
        val path = assetPath(context, emoji) ?: return 0.0
        return kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            com.airbnb.lottie.LottieCompositionFactory.fromAssetSync(context, path).value?.duration?.toDouble()
        } ?: 0.0
    }

    private fun load(context: Context): Map<String, String> = try {
        val text = context.assets.open("$DIR/index.json").bufferedReader().use { it.readText() }
        Json.parseToJsonElement(text).jsonObject.mapValues { it.value.jsonPrimitive.content }
    } catch (_: Exception) {
        emptyMap()
    }
}

suspend fun loadEmojiLoopMs(context: Context, emoji: String): Double = NotoLottie.loadEmojiLoopMsOf(context, emoji)
