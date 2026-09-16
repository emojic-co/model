package ing.emojify.app.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import java.nio.LongBuffer

class OnnxPredictor(assets: AssetManager, meta: Meta) : AutoCloseable {
    data class Prediction(
        val emojiLogits: FloatArray,
        val styleLogits: FloatArray,
        val palettes: List<Palette>,
        val ms: Double,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val char2idx = meta.charToIndex()
    private val session: OrtSession = assets.open("model.onnx").use { stream ->
        env.createSession(stream.readBytes())
    }

    fun predict(text: String, meta: Meta): Prediction {
        val ids = encode(text, meta, char2idx)
        val shape = longArrayOf(1, meta.max_text_len.toLong())
        val t0 = System.nanoTime()
        OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape).use { input ->
            session.run(mapOf("input" to input)).use { result ->
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                @Suppress("UNCHECKED_CAST")
                val emojiLogits = (result.get("emoji_logits").get().value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                val styleLogits = (result.get("style_logits").get().value as Array<FloatArray>)[0]
                @Suppress("UNCHECKED_CAST")
                val colorRows = result.get("color").get().value as Array<FloatArray>
                val colorFlat = FloatArray(colorRows.sumOf { it.size })
                var offset = 0
                for (row in colorRows) {
                    row.copyInto(colorFlat, offset)
                    offset += row.size
                }
                return Prediction(emojiLogits, styleLogits, decodeColorList(colorFlat), ms)
            }
        }
    }

    override fun close() {
        session.close()
    }
}
